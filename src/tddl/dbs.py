from functools import lru_cache

import torch

from tddl.factorizations import factorize_layer
from tddl.utils.model_stats import count_parameters


# factorize_layer = lru_cache(factorize_layer, maxsize=None)
# count_parameters = lru_cache(count_parameters)
# factorize_layer.cache_clear()

@lru_cache
def find_rank_given_error(layer, desired_error, 
    rank = 0.5, 
    tollerance = 0.01, 
    max_rank = 1.0,
    min_rank = 0.0,
    max_iter = 100,
):
    """ given desired error, calculate relative rank """
    for i in range(max_iter):
        # print(rank)
        with torch.no_grad():
            fact_layer, error = factorize_layer(layer, 'tucker', rank, return_error=True)
        # print(error)
        criteria = abs(desired_error - error) > tollerance
        # print(criteria)
        if criteria:
            if error > desired_error: # if the error is too large
                # increase the rank
                old_rank = rank
                rank += (max_rank-rank)/2
                min_rank = old_rank
            else:
                # decrease the rank
                old_rank = rank
                rank -= (rank - min_rank)/2
                max_rank = old_rank
        else:
            break
        # print("-"*10)
    fact_count = count_parameters(fact_layer)
    try:
        print(f"{factorize_layer.cache_info() = }")
    except:
        pass
    try:
        print(f"{count_parameters.cache_info() = }")
    except:
        pass
    return fact_count, rank, error


def compress_layers_with_desired_error(layers, desired_error, baseline_count, *args, **kwargs):
    # fact_layers = []
    ranks = []
    # param_count = 0
    total_fact_count = 0
    for layer in layers:
        # param_count += count_parameters(layers[0][2])
        fact_count, rank, error = find_rank_given_error(layer[2], desired_error = desired_error, *args, **kwargs)
        ranks.append(rank)
        # fact_layers.append(fact_layer)
        total_fact_count += fact_count
    # compression ratio of all layers
    c = total_fact_count/baseline_count
    try:
        print(f"{find_rank_given_error.cache_info() = }")
    except:
        pass
    return c, ranks, error


def find_error_given_c(layers, desired_c, 
    error = 0.5, 
    tollerance = 0.01, 
    max_error = 1.0,
    min_error = 0.0,
    max_iter = 100,
    *args,
    **kwargs,
):
    """ given desired compression rate, calculate relative rank """
    for i in range(max_iter):
        print(error)
        c, ranks, achieved_error = compress_layers_with_desired_error(layers, *args, desired_error=error, tollerance=tollerance, **kwargs)
        print(c)
        print(desired_c - c)
        criteria = abs(desired_c - c) > tollerance
        print(criteria)
        if criteria:
            if c > desired_c: # if the compression rate is too small
                # increase the error
                old_error = error
                error += (max_error-error)/2
                min_error = old_error
            else:
                # decrease the error
                old_error = error
                error -= (error - min_error)/2
                max_error = old_error
        else:
            break
        print("-"*10)
    else:
        print(f"The maximum number of iterations {max_iter} has been reached")
    
    return ranks, c, achieved_error


def undo_factorize_rank_1(layers, ranks):
    #TODO rounding of rank?
    layers = [l for l,r in zip(layers,ranks) if r != 1.0]
    ranks = [r for r in ranks if r != 1.0]

    return layers, ranks
