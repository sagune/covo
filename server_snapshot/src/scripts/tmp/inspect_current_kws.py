import torch, json
path='/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/current_kws/f1G-epoch=7-step=48040.ckpt'
x=torch.load(path,map_location='cpu')
print(json.dumps(x.get('hyper_parameters',{}),ensure_ascii=False,default=str,indent=2))
print('state_keys', list(x.get('state_dict',{}).keys())[:10])
