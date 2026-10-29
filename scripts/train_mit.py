"""Train the fixed full-data or source-group-fold MiT-B2 recipe with MMSeg."""
import train_common
import argparse, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--fold',choices=['primary','secondary'])
 p.add_argument('--full-data',action='store_true')
 p.add_argument('--seed',type=int,choices=[42,43],required=True)
 p.add_argument('--work-dir',type=Path,required=True)
 p.add_argument('--data-root',default='data/raw/UAV-Crack-dataset')
 p.add_argument('--check-only',action='store_true')
 p.add_argument('--resume',action='store_true')
 a=p.parse_args()
 if bool(a.fold)==bool(a.full_data):raise ValueError('choose one of --fold or --full-data')
 fold='full' if a.full_data else a.fold
 manifest=train_common.manifest_for(fold,a.data_root)
 weight=Path('checkpoints/mit_b2_20220624-66e8bf70.pth')
 assert train_common.sha(weight)=='66e8bf7068891148bc097433ddf8e55e690b6f9e17f543edb405f1112b8d330e'
 from mmengine.config import Config
 from mmengine.runner import Runner
 from mmseg.registry import MODELS
 from mmseg.utils import register_all_modules
 cfg=Config.fromfile(str(train_common.ROOT/'configs/uav_crack'/('mit_'+fold+'.py')))
 cfg.launcher='none';cfg.work_dir=str(a.work_dir)
 cfg.randomness=dict(seed=a.seed)
 cfg.optim_wrapper.type='AmpOptimWrapper';cfg.optim_wrapper.loss_scale='dynamic'
 cfg.resume=a.resume
 for key in ('train_dataloader','val_dataloader','test_dataloader'):
  dl=cfg.get(key)
  if dl is not None:
   dl.dataset.ann_file=str(manifest);dl.dataset.data_root=a.data_root
 cfg.model.backbone.init_cfg.checkpoint=str(weight)
 if a.check_only:
  import torch
  register_all_modules()
  torch.manual_seed(a.seed)
  model=MODELS.build(cfg.model)
  model.init_weights();model.cuda().eval()
  from mmengine.registry import DATASETS
  data=DATASETS.build(cfg.train_dataloader.dataset)
  batch=data[0]
  with torch.no_grad():
   inputs=model.data_preprocessor({'inputs':[batch['inputs']],'data_samples':[batch['data_samples']]},True)
   loss=model(**inputs,mode='loss')
  assert all(torch.isfinite(v).all() for v in loss.values() if torch.is_tensor(v))
  print(json.dumps({'ready':True,'fold':fold,'train_count':len(data),'max_iters':cfg.train_cfg.max_iters,'loss':{k:float(v) for k,v in loss.items() if torch.is_tensor(v)}}),flush=True)
  return
 if a.work_dir.exists() and not a.resume:raise FileExistsError(a.work_dir)
 Runner.from_cfg(cfg).train()

if __name__=='__main__':main()
