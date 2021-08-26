# TODO add batch normalization (batch Norm / layer norm () / group norm (Wu and He, 2018))
# https://stackoverflow.com/questions/47143521/where-to-apply-batch-normalization-on-standard-cnns
# TODO max pool after first conv layer
# TODO tune dropout probability
# TODO dropout feature maps

# TODO check which weight initialization is used (Kaiming / MSRA initialization) or Xavier with /2 for relu

import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

from tddl.utils.prime_factors import get_prime_factors


class Net(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64, fc1_out=128):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*12*conv2_out, fc1_out)
        self.fc1_bn = nn.BatchNorm1d(fc1_out)
        self.fc2 = nn.Linear(fc1_out, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_bn(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TdNet(Net):
    def __init__(
        self, conv1_out=32, conv2_out=32, fc1_out=128, 
        layer_nrs=0, rank=0.5, factorization='tucker', td_init=0.02
    ):
        super().__init__(conv1_out=conv1_out, conv2_out=conv2_out, fc1_out=fc1_out)
        
        #TODO this block should be redundant because it should be inherited from Net
        self.conv1 = nn.Conv2d(1, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*12*conv2_out, fc1_out)
        self.fc1_bn = nn.BatchNorm1d(fc1_out)
        self.fc2 = nn.Linear(fc1_out, 10)
        
        # layer_nrs = [2,6] if layer_nr == 0 else [layer_nr]
        for i, (name, module) in enumerate(self.named_modules()):
            if i in layer_nrs:
                if type(module) == torch.nn.modules.conv.Conv2d:
                    fact_layer = tltorch.FactorizedConv.from_conv(
                        module, 
                        rank=rank, 
                        decompose_weights=False, 
                        factorization=factorization
                    )
                elif type(module) == torch.nn.modules.linear.Linear:
                    fact_layer = tltorch.FactorizedLinear.from_linear(
                        module, 
                        in_tensorized_features=get_prime_factors(module.in_features), 
                        out_tensorized_features=get_prime_factors(module.out_features), 
                        rank=rank,
                        factorization=factorization,
                    )
                if td_init:
                    fact_layer.weight.normal_(0, td_init)
                self._modules[name] = fact_layer

    