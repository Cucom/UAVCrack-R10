"""Run the adapted SMP official binary example with Lightning's training loop."""
import train_common
import argparse, hashlib, json
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from projects.uav_crack.resunet_description import CrackDataset, DescriptionResUNet


class EpochLog(Callback):
    def on_train_epoch_end(self, trainer, module):
        print(json.dumps({'epoch':trainer.current_epoch+1,'global_step':trainer.global_step,
           'metrics':{k:float(v) for k,v in trainer.callback_metrics.items()},
           'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=[42,43],required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('configs/uav_crack/resunet_description_v1.json'))
    p.add_argument('--data-root',default='data/raw/UAV-Crack-dataset')
    p.add_argument('--full-data',action='store_true')
    p.add_argument('--fold', choices=['primary','secondary'], default='primary')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--epochs-limit',type=int)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    if not a.smoke:
        if a.epochs_limit:
            raise ValueError('formal budget fixed at 42 epochs; epochs-limit is smoke only')
    split=train_common.manifest_for('full' if a.full_data else a.fold, a.data_root)
    weight=Path(protocol['pretrained'])
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==protocol['pretrained_sha256']
    if a.work_dir.exists() and not a.resume:
        raise FileExistsError('refuse to overwrite work directory')
    pl.seed_everything(a.seed,workers=True)
    torch.backends.cudnn.benchmark=False
    train=CrackDataset(a.data_root,split,'train',True,16 if a.smoke else 0)
    val=CrackDataset(a.data_root,split,'val',False,12 if a.smoke else 0)
    if not a.full_data:
        assert {r['source_group'] for r in train.rows}.isdisjoint({r['source_group'] for r in val.rows})
    settings=protocol['implementation_assumptions_not_reported_by_author']
    loader_args={'batch_size':settings['batch_size'],'num_workers':settings['num_workers'],'pin_memory':True}
    train_dl=DataLoader(train,shuffle=True,**loader_args)
    val_dl=None if a.full_data else DataLoader(val,shuffle=False,**loader_args)
    epochs=a.epochs_limit or (2 if a.smoke else protocol['known_from_pdf']['epochs'])
    run={'protocol':protocol,'seed':a.seed,'split':'full' if a.full_data else a.fold,
         'train_count':len(train),'validation_count':0 if a.full_data else len(val),
         'epochs':epochs,'steps_per_epoch':len(train_dl),'smoke':a.smoke}
    if a.check_only:
        model=DescriptionResUNet(protocol,initialize=True).cuda().eval()
        batch=train[0]
        with torch.no_grad():
            logits=model(batch['image'][None].cuda())
            loss=model.loss_fn(logits,batch['mask'][None].cuda())
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        run['check_loss']=float(loss)
        print(json.dumps({'check_only':True,'ready':True,'run':run}),flush=True)
        return
    a.work_dir.mkdir(parents=True,exist_ok=True)
    (a.work_dir/'actual_config.json').write_text(json.dumps(run,indent=2)+'\n')
    print(json.dumps(run),flush=True)
    checkpoint=ModelCheckpoint(dirpath=a.work_dir/'checkpoints',filename='epoch{epoch:02d}-step{step}',
         monitor=None if a.full_data else 'valid_dataset_iou',mode='max',save_top_k=0 if a.full_data else 1,
         save_last=True,every_n_epochs=1)
    model=DescriptionResUNet(protocol,initialize=not bool(a.resume))
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision=settings['precision'],max_epochs=epochs,
       callbacks=[checkpoint,EpochLog()],logger=CSVLogger(a.work_dir/'csv',name='metrics'),
       log_every_n_steps=1 if a.smoke else 20,num_sanity_val_steps=0,enable_progress_bar=False,
       deterministic='warn')
    trainer.fit(model,train_dl,val_dl,ckpt_path=str(a.resume) if a.resume else None)
    print(json.dumps({'finished':True,'epoch':trainer.current_epoch,'global_step':trainer.global_step,
        'last':checkpoint.last_model_path,'best':checkpoint.best_model_path,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)


if __name__=='__main__':
    main()
