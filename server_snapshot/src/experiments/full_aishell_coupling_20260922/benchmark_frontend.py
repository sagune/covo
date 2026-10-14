"""Bounded training-only throughput benchmark; no train/dev metrics or training."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path[:0] = [str(SRC), str(SRC / 'data')]
os.chdir(SRC)
os.environ.update(CBW_DEBUG_LOG='off', CBW_DEBUG_CLEAR_ON_START='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
from prepare_full import legacy, DATA


def main():
    import fcntl
    import subprocess
    import torch
    import yaml
    from model.cb_sensevoice import CBSenseVoice
    p = legacy()
    with (HERE.parent / 'text_residual_coupling_gpu.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        torch.set_num_threads(6)
        args = yaml.safe_load((SRC / 'configs/cb-sensevoice-aishell.yaml').read_text())['model']['init_args']
        args.update(split='train_covo_full', oracle_nbest_diagnostic=False, sensevoice_ckpt=str(p.SENSE))
        # All 20000 training keywords, identical for every row; no reference routing.
        model = CBSenseVoice(**args).cuda().eval()
        rows = sorted(p.load(DATA / 'train.manifest.jsonl'), key=lambda r: p.rank(r['id']))[:3]
        records = []
        with torch.inference_mode():
            for row in rows:
                start = time.time()
                prediction = model._forward_sensevoice(row['wav'], [])
                record = model._build_covo_evidence_record({'id':row['id']}, 0, prediction, [], [])
                records.append(dict(id=row['id'], seconds=time.time()-start, prediction=prediction,
                                    evidence=record['input']))
                p.save(HERE / 'frontend_benchmark.partial.json', dict(context_keywords=len(model.kw_database), records=records))
                print(json.dumps(dict(id=row['id'], seconds=records[-1]['seconds'], context_keywords=len(model.kw_database))), flush=True)
        p.save(HERE / 'frontend_benchmark.json', dict(status='COMPLETE', context_keywords=len(model.kw_database),
               kws_semantics='online normalized float32 withitn frames, not old quantized cache; benchmark only',
               config=args, records=records, model_source_sha256=p.digest(SRC / 'model/cb_sensevoice.py')))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        legacy().save(HERE / 'frontend_benchmark_failure.json', dict(error=traceback.format_exc(), timestamp=time.time()))
        raise
