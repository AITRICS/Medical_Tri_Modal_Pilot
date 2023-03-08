import torch.nn as nn
from monai.networks.nets import ViT


class VIT(nn.Module):
    def __init__(self, args):
        super(VIT, self).__init__()
        self.vit     = ViT(in_channels=1, img_size=(224,224), 
                           patch_size=args.vit_patch_size, spatial_dims=2,
                           hidden_size=256, mlp_dim=1024,
                           num_heads = 4, num_layers=args.vit_num_layers, 
                           classification=True, num_classes=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y_hat, _ = self.vit(x)
        return self.sigmoid(y_hat)
