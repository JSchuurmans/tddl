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
    def __init__(self, in_channels=1, conv1_out=32, conv2_out=64, fc1_out=128, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*12*conv2_out, fc1_out) #TODO: does 12 by 12 match for Fashion-MNIST and CIFAR-10?
        self.fc1_bn = nn.BatchNorm1d(fc1_out)
        self.fc2 = nn.Linear(fc1_out, num_classes)

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
        self, in_channels=1, conv1_out=32, conv2_out=32, fc1_out=128, num_classes=10,
        layer_nrs=0, rank=0.5, factorization='tucker', td_init=0.02
    ):
        super().__init__(conv1_out=conv1_out, conv2_out=conv2_out, fc1_out=fc1_out)
        
        #TODO this block should be redundant because it should be inherited from Net
        self.conv1 = nn.Conv2d(in_channels, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*12*conv2_out, fc1_out)
        self.fc1_bn = nn.BatchNorm1d(fc1_out)
        self.fc2 = nn.Linear(fc1_out, num_classes)

        decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
        fixed_rank_modes = 'spatial' if factorization == 'tucker' else None
        
        # layer_nrs = [2,6] if layer_nr == 0 else [layer_nr]
        for i, (name, module) in enumerate(self.named_modules()):
            if i in layer_nrs:
                if type(module) == torch.nn.modules.conv.Conv2d:
                    fact_layer = tltorch.FactorizedConv.from_conv(
                        module, 
                        rank=rank, 
                        decompose_weights=False, 
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        decomposition_kwargs=decomposition_kwargs,
                    )
                elif type(module) == torch.nn.modules.linear.Linear:
                    # fact_layer = tltorch.FactorizedLinear(
                    #     in_tensorized_features=get_prime_factors(module.in_features), 
                    #     out_tensorized_features=get_prime_factors(module.out_features), 
                    #     rank=rank,
                    #     factorization=factorization,
                    #     device=module.weight.device, #  <-- this gives me errors
                    # )
                    
                    fact_layer = tltorch.FactorizedLinear.from_linear(
                        module, 
                        in_tensorized_features=get_prime_factors(module.in_features), 
                        out_tensorized_features=get_prime_factors(module.out_features), 
                        rank=rank,
                        factorization=factorization,
                        decomposition_kwargs=decomposition_kwargs,
                    )
                if td_init:
                    fact_layer.weight.normal_(0, td_init)
                self._modules[name] = fact_layer


class TensorNet(nn.Module):
    def __init__(self, tcl_rank, in_channels, num_classes, conv1_out=32, conv2_out=64, fc1_out=12):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = tltorch.factorized_layers.TLC(12*12*conv2_out, tcl_rank, fc1_out)
        self.fc1_bn = nn.BatchNorm1d(fc1_out)
        self.fc2 = tltorch.factorized_layers.TRL(fc1_out, num_classes, bias=True) # bias for linear layer is true by default

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


class GaripovNet(nn.Module):
    def __init__(
        self, in_channels, num_classes, conv1_out=64, conv2_out=64, conv3_out=128, conv4_out=128, conv5_out=128, conv6_out=128, fc_in=128,
    ):
        super(GaripovNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, 3, 1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(conv1_out)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, 3, 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(conv3_out)
        self.conv4 = nn.Conv2d(conv3_out, conv4_out, 3, 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(conv4_out)
        self.conv5 = nn.Conv2d(conv4_out, conv5_out, 3, 1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(conv5_out)
        self.conv6 = nn.Conv2d(conv5_out, conv6_out, 3, 1, padding=1)
        self.fc1 = nn.Linear(fc_in, num_classes) # TODO: change 12 by 12 # TODO: check num_classes
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = F.avg_pool2d(x, kernel_size=4)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        output = F.log_softmax(x, dim=1) # TODO: check if needed
        return output


class JaderbergNet(nn.Module):
    def __init__(
        self, in_channels, num_classes, conv1_out=96, conv2_out=128, conv3_out=512, conv4_out=148,
        channel_max_pool_kernels=(2,2,4,4)
    ):
        super(JaderbergNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_out, 9)
        self.conv2 = nn.Conv2d(
            int(conv1_out/channel_max_pool_kernels[0]), conv2_out, 9,
        )
        self.conv3 = nn.Conv2d(
            int(conv2_out/channel_max_pool_kernels[1]), conv3_out, 8,
            )
        self.conv4 = nn.Conv2d(
            int(conv3_out/channel_max_pool_kernels[2]), 
            num_classes*channel_max_pool_kernels[-1], 
            1,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = channel_max_pool(x, kernel_size=(2,1,1))

        x = self.conv2(x)
        x = channel_max_pool(x, kernel_size=(2,1,1))

        x = self.conv3(x)
        x = channel_max_pool(x, kernel_size=(4,1,1))

        x = self.conv4(x)
        x = channel_max_pool(x, kernel_size=(4,1,1))

        x = torch.flatten(x, 1)
        output = F.log_softmax(x, dim=1)

        return output

def channel_max_pool(x, *kwargs):
    return F.max_pool3d(x[:,None,...], *kwargs)[:,0,...]