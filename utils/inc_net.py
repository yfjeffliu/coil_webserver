import copy
import torch
from torch import nn

from convs.ucir_resnet import resnet18 as cosine_resnet18

from convs.linears import  CosineLinear


def get_convnet(convnet_type, pretrained=False):
    return cosine_resnet18(pretrained=pretrained)



class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    # def update_fc_fixed(self, nb_classes):
    #     pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self




class SimpleCosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:

                weight = torch.cat([weight.cuda(), nextperiod_initialization.cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        # print(in_dim, out_dim)
        fc = CosineLinear(in_dim, out_dim)
        return fc


# class BiasLayer(nn.Module):
#     def __init__(self):
#         super(BiasLayer, self).__init__()
#         self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
#         self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

#     def forward(self, x, low_range, high_range):
#         ret_x = x.clone()
#         ret_x[:, low_range:high_range] = self.alpha * \
#             x[:, low_range:high_range] + self.beta
#         return ret_x

#     def get_params(self):
#         return (self.alpha.item(), self.beta.item())

