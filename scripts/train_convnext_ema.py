"""Train ConvNeXt-Tiny with a separately maintained EMA shadow model."""
import train_common
import argparse, csv, copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from projects.uav_crack.resunet_description import CrackDataset
from projects.uav_crack.convnext_unet import ConvNeXtUNet


class LitConvNeXtEMA(pl.LightningModule):
 def __init__(self,decay=.999,pretrained=True,pretrained_path=None):
  super().__init__();self.net=ConvNeXtUNet(pretrained=pretrained,pretrained_path=pretrained_path)
  self.ema_net=copy.deepcopy(self.net).eval()
  for p in self.ema_net.parameters():p.requires_grad_(False)
  self.decay=decay;self.ema_updates=0
  self.save_hyperparameters({'decay':decay,'pretrained':False})
 def train(self,mode=True):
  super().train(mode);self.ema_net.eval();return self
 def training_step(self,batch,batch_idx):
  loss,logits=self.net.training_loss(batch['image'],batch['mask'])
  pred=logits.sigmoid()>=.4;truth=batch['mask'].bool()
  tp=(pred&truth).sum();fp=(pred&~truth).sum();fn=(~pred&truth).sum()
  self.log('train_loss_step',loss.detach(),on_step=True,on_epoch=False)
  self.log('train_f1_step',2*tp/(2*tp+fp+fn).clamp_min(1),on_step=True,on_epoch=False)
  return loss
 @torch.no_grad()
 def on_train_batch_end(self,outputs,batch,batch_idx):
  src=dict(self.net.named_parameters());ema=dict(self.ema_net.named_parameters())
  for name,target in ema.items():target.mul_(self.decay).add_(src[name],alpha=1-self.decay)
  srcb=dict(self.net.named_buffers());emab=dict(self.ema_net.named_buffers())
  for name,target in emab.items():
   value=srcb[name]
   if target.is_floating_point():target.mul_(self.decay).add_(value,alpha=1-self.decay)
   else:target.copy_(value)
  self.ema_updates+=1
 def configure_optimizers(self):return self.net.configure_optimizers()


class EpochLog(Callback):
 def on_train_epoch_end(self,trainer,module):
  print(json.dumps({'epoch':trainer.current_epoch+1,'global_step':trainer.global_step,
    'ema_updates':module.ema_updates,'metrics':{k:float(v) for k,v in trainer.callback_metrics.items()},
    'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--fold',choices=['primary','secondary'])
 p.add_argument('--full-data',action='store_true')
 p.add_argument('--seed',type=int,choices=[42,43],required=True)
 p.add_argument('--work-dir',type=Path,required=True)
 p.add_argument('--data-root',default='data/raw/UAV-Crack-dataset')
 p.add_argument('--check-only',action='store_true')
 a=p.parse_args()
 if bool(a.fold)==bool(a.full_data):raise ValueError('choose exactly one of --fold or --full-data')
 fold='full' if a.full_data else a.fold
 manifest=train_common.manifest_for(fold,a.data_root)
 weight=Path('checkpoints/convnext_tiny_1k_224_ema.pth')
 assert sha(weight)=='14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595'
 pl.seed_everything(a.seed,workers=True);torch.backends.cudnn.benchmark=False
 ds=CrackDataset(a.data_root,manifest,'train',True)
 dl=DataLoader(ds,batch_size=8,shuffle=True,num_workers=2,pin_memory=True)
 model=LitConvNeXtEMA(decay=.999,pretrained=True,pretrained_path=weight)
 run={'fold':fold,'seed':a.seed,'manifest_sha256':sha(manifest),
  'pretrained_sha256':sha(weight),'ema_decay':.999,'batch_size':8,
  'train_images':len(ds),'steps_per_epoch':len(dl),'epochs':42,
  'validation_during_training':False}
 if a.check_only:
  model.cuda().eval()
  batch=ds[0]
  with torch.no_grad():loss,logits=model.net.training_loss(batch['image'][None].cuda(),batch['mask'][None].cuda())
  assert torch.isfinite(logits).all() and torch.isfinite(loss)
  print(json.dumps({'check_only':True,'ready':True,'run':run,'loss':float(loss)}),flush=True);return
 work=a.work_dir
 if work.exists():raise FileExistsError(work)
 work.mkdir(parents=True)
 (work/'run.json').write_text(json.dumps(run,indent=2)+'\n')
 ckpt=ModelCheckpoint(dirpath=work/'checkpoints',filename='epoch{epoch:02d}-step{step}',monitor=None,
  save_top_k=0,save_last=True,every_n_epochs=1)
 trainer=pl.Trainer(accelerator='gpu',devices=1,precision='16-mixed',max_epochs=42,
  callbacks=[ckpt,EpochLog()],logger=CSVLogger(work/'csv',name='metrics'),log_every_n_steps=20,
  num_sanity_val_steps=0,enable_progress_bar=False,deterministic='warn')
 started=time.monotonic();trainer.fit(model,dl)
 final=work/'checkpoints/ema_last.pth'
 torch.save({'state_dict':model.ema_net.state_dict(),'ema_updates':model.ema_updates,
  'decay':model.decay,'epoch':trainer.current_epoch-1,'global_step':trainer.global_step},final)
 print(json.dumps({'finished':True,'fold':fold,'seed':a.seed,'global_step':trainer.global_step,
  'ema_updates':model.ema_updates,'last_ema':str(final),
  'elapsed_seconds':time.monotonic()-started},indent=2),flush=True)

if __name__=='__main__':main()
