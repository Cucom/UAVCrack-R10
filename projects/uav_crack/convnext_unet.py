"""ConvNeXt-Tiny U-Net screen with a fixed ImageNet-1K timm encoder."""
import torch
from torch.nn import functional as F
from projects.uav_crack.resunet_description import BCEDice, CrackDataset
import segmentation_models_pytorch as smp


class ConvNeXtUNet(torch.nn.Module):
    def __init__(self, pretrained=False, pretrained_path=None):
        super().__init__()
        kwargs = {'pretrained_cfg_overlay': {'file': str(pretrained_path)}} if pretrained_path else {}
        self.model = smp.Unet(encoder_name='tu-convnext_tiny.fb_in1k',
                              encoder_weights='imagenet' if pretrained else None,
                              in_channels=3, classes=1, activation=None, **kwargs)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1,3,1,1))
        self.loss_fn = BCEDice()

    def forward(self, image):
        h, w = image.shape[-2:]
        image = (image-self.mean)/self.std
        padded = F.pad(image, (0, (-w)%32, 0, (-h)%32))
        return self.model(padded)[..., :h, :w]

    def training_loss(self, image, mask):
        logits = self(image)
        return self.loss_fn(logits, mask), logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': 2e-5},
            {'params': list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters()), 'lr': 3e-4},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=42, eta_min=1e-6)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
