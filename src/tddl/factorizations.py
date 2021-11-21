import torch
import tltorch
from tddl.utils.prime_factors import get_prime_factors
from tddl.utils.approximation import calculate_error

def factorize_layer(
    module,
    factorization='tucker',
    rank=0.5,
    decompose_weights=True,
    # fixed_rank_modes=None,
    # decomposition_kwargs={},
    init_std=None,
    return_error=False,
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
        )
    else:
        raise NotImplementedError(type(module))

    if init_std:
        fact_module.weight.normal_(0, init_std)

    if return_error:
        error = calculate_error(fact_module.weight.to_tensor(), module.weight)
    else:
        error = None

    return fact_module, error


def factorize_network(
    model, layers=[], exclude=[], verbose=False, 
    return_error=False, 
    **kwargs
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
        
        if children == {}:
            return m
        else:
        # look for children from children... to the last child!
            for name, child in children.items():
                if verbose:
                    print(i, name, type(child))
                if name in exclude:
                    i+=1
                    continue
                # if type(child) == torch.nn.modules.conv.Conv2d and i in layers:
                if i in layers or name in layers or type(child) in layers:
                    # if return_error:
                    #     layer, error = factorize_layer(child, **kwargs)
                    #     if verbose:
                    #         print(error)
                    # else:
                    #     layer = 
                    m._modules[name], error = factorize_layer(child, return_error=return_error, **kwargs)
                try:
                    if verbose and return_error:
                        print((i, error))
                    output[name] = (i, error, nested_children(child, **kwargs) )
                except TypeError:
                    output[name] = (i, error, nested_children(child, **kwargs) )
        return output #, errors
    out = nested_children(model, **kwargs)

    if return_error:
        return out

def number_layers(model, verbose=False, **kwargs):
    i = -1
    def nested_children(m: torch.nn.Module, exclude=[]):
        nonlocal i
        children = dict(m.named_children())
        output = {}
        i+=1
        if children == {}:
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


def outer(output, layers):
    list_errors = []

    def parse_errors(d, layers):
        
        nonlocal list_errors
        for k, v in d.items():
            if isinstance(v[2], dict):
                parse_errors(v[2], layers)
            elif v[0] in layers:
                list_errors.append(v)

    parse_errors(output, layers)
    
    return list_errors