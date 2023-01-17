from functools import lru_cache

import torch
import tltorch

from tddl.utils.prime_factors import get_prime_factors
from tddl.utils.approximation import calculate_relative_error


RESNET_LAYERS = [0,6,9,12,15,19,22,25,28,31,35,38,41,44,47,51,54,57,60,63,66]


@lru_cache(maxsize=4096)
def factorize_layer(
    module,
    factorization='tucker',
    rank=0.5,
    decompose_weights=True,
    init_std=None,
    return_error=False,
    **kwargs,
):
    
    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None

    if type(module) == torch.nn.modules.conv.Conv2d:
        fact_module = tltorch.FactorizedConv.from_conv(
            module, 
            rank=rank, 
            decompose_weights=decompose_weights, 
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs,
            **kwargs,
        )
    elif type(module) == torch.nn.modules.linear.Linear:
        fact_module = tltorch.FactorizedLinear.from_linear(
            module,
            in_tensorized_features=get_prime_factors(module.in_features),
            out_tensorized_features=get_prime_factors(module.out_features),
            rank=rank,
            factorization=factorization,
            decompose_weights=decompose_weights,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs,
            **kwargs,
        )
    else:
        raise NotImplementedError(type(module))

    if init_std:
        fact_module.weight.normal_(0, init_std)

    if return_error:
        error = calculate_relative_error(
            original=module.weight,
            approximation=fact_module.weight.to_tensor(),
        )
        return fact_module, error
        
    else:
        return fact_module


def factorize_network(
    model, layers=[], exclude=[], verbose=False, 
    return_error=False, 
    **kwargs,
):
    """
    Usage: factorize_network(model, layers=[6])
    """
    i = -1
    def nested_children(m: torch.nn.Module, **kwargs):
        """
        layers: List of either
            layer numbers (numbered according to number_layers(model))
            layer names
            module types
        """
        nonlocal i
        children = dict(m.named_children())
        output = {}
        error = None

        i+=1
        
        if children == {} or m._get_name()=='FactorizedConv':
            return m
        else:
        # look for children from children... to the last child!
            for name, child in children.items():
                if verbose:
                    print(i, name, type(child))
                if name in exclude:
                    i+=1
                    continue
                if i in layers or name in layers or type(child) in layers:
                    m._modules[name] = factorize_layer(child, return_error=return_error, **kwargs)
                try:
                    output[name] = (i, nested_children(child, **kwargs) )
                except TypeError:
                    output[name] = (i, nested_children(child, **kwargs) )
        return output
    out = nested_children(model, **kwargs)

    if return_error:
        return out


def number_layers(model, verbose=False, end_nodes=[], **kwargs):
    i = -1
    def nested_children(m: torch.nn.Module, exclude=[]):
        nonlocal i
        children = dict(m.named_children())
        output = {}
        i+=1
        if children == {} or m._get_name()=='FactorizedConv':
            return m
        else:
            for name, child in children.items():
                if verbose:
                    print(i, name, type(child))
                if name in exclude:
                    i+=1
                    continue
                try:
                    output[name] = (i,nested_children(child, exclude=exclude))
                except TypeError:
                    output[name] = (i,nested_children(child, exclude=exclude))
        return output
    return nested_children(model, **kwargs)


def list_errors(output, layers):
    errors = []

    def parse_errors(d, layers):
        
        nonlocal errors
        for k, v in d.items():
            if isinstance(v[2], dict):
                parse_errors(v[2], layers)
            elif v[0] in layers:
                errors.append(
                    (
                        v[0], # layer_nr
                        float(v[1].detach().cpu()), # approx.error wrt pretrained
                        str(v[2]), # layer
                    )
                )

    parse_errors(output, layers)
    
    return errors


def listify_numbered_layers(numbered_layers, layer_names=[], layer_nrs=[]):
    output = []

    def parse_errors(d):
        
        nonlocal output
        for k, v in d.items():
            # print(v)
            if isinstance(v[1], dict):
                parse_errors(v[1])
            elif k in layer_names or v[0] in layer_nrs:
                output.append((
                        k, # layer_name
                        v[0], # layer_nr
                        v[1], # layer
                    ))
    
    parse_errors(numbered_layers)
    
    return output


def get_weights(model, layer_nrs=RESNET_LAYERS):
    numbered_layers = number_layers(model)
    listed_layers = listify_numbered_layers(numbered_layers=numbered_layers, layer_nrs=layer_nrs)
    return {nr:layer.weight for name,nr,layer in listed_layers}


def factorize_network_different_ranks(model, layers, ranks, **kwargs):
    
    for layer_nr, rank in zip(layers, ranks):
        factorize_network(model, layers=[layer_nr], rank=rank, **kwargs)
