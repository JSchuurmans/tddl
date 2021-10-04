import copy
import tltorch
from tddl.models.resnet import PA_ResNet18
from tddl.utils.prime_factors import get_prime_factors

def low_rank_resnet18(layers, rank=0.5, decompose_weights=False, factorization='tucker', init=None, pretrained_model=None, **kwargs):
    if pretrained_model is None:
        model = PA_ResNet18(**kwargs)
    else:
        model = pretrained_model.cpu()

    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None
    
    if 0 in layers:
        model.conv1 = tltorch.FactorizedConv.from_conv(
                model.conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init is not None:
            model.conv1.weight.normal_(0, init)

    # layer1
    if 1 in layers:
        model.layer1[0].conv1  = tltorch.FactorizedConv.from_conv(
                model.layer1[0].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer1[0].conv1.weight.normal_(0, init)
    if 2 in layers:
        model.layer1[0].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer1[0].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer1[0].conv2.weight.normal_(0, init)
    if 3 in layers:
        model.layer1[1].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer1[1].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer1[1].conv1.weight.normal_(0, init)
    if 4 in layers:
        model.layer1[1].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer1[1].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer1[1].conv2.weight.normal_(0, init)

    # layer2
    if 5 in layers:
        model.layer2[0].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer2[0].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer2[0].conv1.weight.normal_(0, init)
    if 6 in layers:
        model.layer2[0].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer2[0].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer2[0].conv2.weight.normal_(0, init)
    if 7 in layers:
        model.layer2[1].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer2[1].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer2[1].conv1.weight.normal_(0, init)
    if 8 in layers:
        model.layer2[1].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer2[1].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer2[1].conv2.weight.normal_(0, init)

    # layer3
    if 9 in layers:
        model.layer3[0].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer3[0].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer3[0].conv1.weight.normal_(0, init)
    if 10 in layers:
        model.layer3[0].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer3[0].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer3[0].conv2.weight.normal_(0, init)
    if 11 in layers:
        model.layer3[1].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer3[1].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer3[1].conv1.weight.normal_(0, init)
    if 12 in layers:
        model.layer3[1].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer3[1].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer3[1].conv2.weight.normal_(0, init)

    # layer4
    if 13 in layers:
        model.layer4[0].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer4[0].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer4[0].conv1.weight.normal_(0, init)
    if 14 in layers:
        model.layer4[0].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer4[0].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer4[0].conv2.weight.normal_(0, init)
    if 15 in layers:
        model.layer4[1].conv1 = tltorch.FactorizedConv.from_conv(
                model.layer4[1].conv1, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer4[1].conv1.weight.normal_(0, init)
    if 16 in layers:
        model.layer4[1].conv2 = tltorch.FactorizedConv.from_conv(
                model.layer4[1].conv2, 
                rank=rank, 
                decompose_weights=decompose_weights, 
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                fixed_rank_modes=fixed_rank_modes,
            )
        if init:
            model.layer4[1].conv2.weight.normal_(0, init)

    # linear
    if 17 in layers:
        module = model.linear
        model.linear = tltorch.FactorizedLinear.from_linear(
            module, 
            in_tensorized_features=get_prime_factors(module.in_features), 
            out_tensorized_features=get_prime_factors(module.out_features), 
            rank=rank,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
        )
        if init:
            model.conv1.weight.normal_(0, init)

    return model