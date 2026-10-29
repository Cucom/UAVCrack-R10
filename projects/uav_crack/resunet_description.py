"""Public-description reproduction using SMP's official binary example.

Network and training loop remain SMP Unet and Lightning Trainer respectively.
This module only adapts data, valid-area handling, loss, optimizer and metrics.
"""
import csv
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from projects.uav_crack.smp_binary_example_v050 import PetModel
import segmentation_models_pytorch as smp


class CrackDataset(Dataset):
    def __init__(self, root, manifest, split, augment=False, limit=0):
        self.root = Path(root)
        with Path(manifest).open(newline='', encoding='utf-8') as f:
            self.rows = [r for r in csv.DictReader(f) if split == 'all' or r['split'] == split]
        if limit:
            self.rows = self.rows[:limit]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(self.root/r['image']) as im:
            x = np.asarray(im.convert('RGB'), dtype=np.float32)/255.
        with Image.open(self.root/r['mask']) as im:
            y = np.array(im, dtype=np.uint8)
        if x.shape[:2] != (378,672) or y.shape != (378,672) or not np.isin(y,[0,1]).all():
            raise ValueError(f'bad image/mask pair: {r}')
        if self.augment:
            for axis in (0,1):
                if torch.rand(()) < 0.5:
                    x, y = np.flip(x,axis), np.flip(y,axis)
        return {'image': torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))),
                'mask': torch.from_numpy(np.ascontiguousarray(y[None])).float(),
                'index': index}


class BCEDice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True, smooth=0., eps=1e-7)

    def forward(self, logits, target):
        # Inputs contain only the valid image area; padded rows never enter loss.
        return F.binary_cross_entropy_with_logits(logits, target) + self.dice(logits,target)


class DescriptionResUNet(PetModel):
    def __init__(self, protocol, initialize=True):
        super().__init__('Unet', 'resnet34', in_channels=3, out_classes=1, encoder_weights=None)
        self.protocol = protocol
        self.save_hyperparameters({'protocol':protocol, 'initialize':False})
        if initialize:
            weights = torch.load(protocol['pretrained'],map_location='cpu',weights_only=True)
            self.model.encoder.load_state_dict(weights, strict=True)
        self.loss_fn = BCEDice()
        self.threshold = protocol['known_from_pdf']['threshold']

    def forward(self, image):
        # Normalize before padding, as used by the existing project protocol.
        h,w = image.shape[-2:]
        normalized = (image-self.mean)/self.std
        padded = F.pad(normalized,(0,(-w)%32,0,(-h)%32))
        return self.model(padded)[...,:h,:w]

    def shared_step(self, batch, stage):
        image, mask = batch['image'], batch['mask']
        assert mask.min() >= 0 and mask.max() <= 1
        logits = self(image)
        loss = self.loss_fn(logits, mask)
        pred = logits.sigmoid() >= self.threshold
        tp,fp,fn,tn = smp.metrics.get_stats(pred.long(), mask.long(), mode='binary')
        return {'loss':loss,'tp':tp,'fp':fp,'fn':fn,'tn':tn}

    def shared_epoch_end(self, outputs, stage):
        # Retain the official example's dataset-wise/image-wise IoU calculation.
        super().shared_epoch_end(outputs,stage)
        tp,fp,fn,tn = [torch.cat([o[k] for o in outputs]).sum().float() for k in ('tp','fp','fn','tn')]
        fg = tp/(tp+fp+fn).clamp_min(1)
        bg = tn/(tn+fp+fn).clamp_min(1)
        self.log_dict({f'{stage}_mIoU':(fg+bg)/2,
                       f'{stage}_F1':2*tp/(2*tp+fp+fn).clamp_min(1),
                       f'{stage}_loss':torch.stack([o['loss'].detach() for o in outputs]).mean()},prog_bar=False)

    def configure_optimizers(self):
        a = self.protocol['implementation_assumptions_not_reported_by_author']
        optimizer = torch.optim.AdamW(self.parameters(),lr=a['lr'],weight_decay=a['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=self.protocol['known_from_pdf']['epochs'],eta_min=a['eta_min'])
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'epoch'}}
