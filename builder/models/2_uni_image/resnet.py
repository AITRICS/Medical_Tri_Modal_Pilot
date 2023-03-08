import torch.nn as nn
from monai.networks.nets.resnet import ResNet, ResNetBlock, ResNetBottleneck

class RESNET(nn.Module):
    def __init__(self, args):
        super(RESNET, self).__init__()
        print("### preparing resnet model ###")
        if args.resnet_num_layers == 18:
            self.resnet = ResNet(block=ResNetBlock, layers=[2,2,2,2], block_inplanes=[32,64,128,256],
                                 spatial_dims=2, conv1_t_stride=2, n_input_channels=1, num_classes=1)
        elif args.resnet_num_layers == 34:
            self.resnet = ResNet(block=ResNetBlock, layers=[3,4,6,3], block_inplanes=[32,64,128,256],
                                 spatial_dims=2, conv1_t_stride=2, n_input_channels=1, num_classes=1)
        elif args.resnet_num_layers == 50:
            self.resnet = ResNet(block=ResNetBottleneck, layers=[3,4,6,3], block_inplanes=[32,64,128,256],
                                 spatial_dims=2, conv1_t_stride=2, n_input_channels=1, num_classes=1)
        else:
            raise NotImplementedError('resnet: 18,34,50')
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y_hat = self.resnet(x)
        return self.sigmoid(y_hat)

