"""Continue existing LoRA for one epoch; assistant-only loss, no silent truncation."""
import json
import argparse
import math
from pathlib import Path
import time
from prepare import OUT,MODEL,ADAPTER
DATA_OUT=OUT


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=OUT)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--learning-rate',type=float,default=2e-5)
    parser.add_argument('--seed',type=int,default=20260908)
    config=parser.parse_args()
    OUT=config.run_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    steps=625*config.epochs
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
    from peft import PeftModel
    set_seed(config.seed)
    target=OUT/'adapter'
    if target.exists():raise FileExistsError('Never overwrite adapter')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    features=[]
    for line in (DATA_OUT/'train.unified.jsonl').read_text().splitlines():
        r=json.loads(line);messages=r['messages']
        prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        pi=tokenizer(prefix,add_special_tokens=False)['input_ids'];fi=tokenizer(full,add_special_tokens=False)['input_ids']
        assert fi[:len(pi)]==pi,'Chat template prefix mismatch'
        assert len(pi)<len(fi)<=2048
        features.append({'input_ids':fi,'attention_mask':[1]*len(fi),'labels':[-100]*len(pi)+fi[len(pi):]})
    assert len(features)==5000
    dump('loss_mask_audit.json',dict(samples=5000,all_prompt_tokens_masked=True,min_target_tokens=min(sum(x!=-100 for x in r['labels']) for r in features)))
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='auto')
    model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=True)
    model.config.use_cache=False
    model.enable_input_require_grads()
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n in trainable)
    model.print_trainable_parameters()
    args=TrainingArguments(output_dir=str(target),num_train_epochs=config.epochs,learning_rate=config.learning_rate,
        per_device_train_batch_size=2,gradient_accumulation_steps=4,bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        warmup_ratio=.03,logging_steps=10,save_strategy='steps',save_steps=math.ceil(steps/2),
        save_only_model=True,report_to='none',seed=config.seed,data_seed=config.seed,
        dataloader_num_workers=0,lr_scheduler_type='linear')
    dump('training_plan.json',dict(epochs=config.epochs,rows=5000,effective_batch=8,optimizer_steps=steps,learning_rate=config.learning_rate,
         seed=config.seed,checkpoints=[math.ceil(steps/2),steps],completion_only=True,base_frozen=True,starting_adapter=str(ADAPTER)))
    trainer=Trainer(model=model,args=args,train_dataset=Dataset.from_list(features),
                    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100))
    started=time.time();trainer.train()
    final=target/f'checkpoint-{steps}';trainer.save_model(str(final));tokenizer.save_pretrained(str(final))
    dump('TRAIN_DONE.json',dict(elapsed_seconds=time.time()-started,final=str(final),steps=trainer.state.global_step))


if __name__=='__main__':main()
