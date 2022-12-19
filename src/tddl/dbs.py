import torch

from tddl.factorizations import factorize_layer
from tddl.utils.model_stats import count_parameters


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
    return fact_layer, rank, error


def compress_layers_with_desired_error(layers, desired_error, baseline_count, *args, **kwargs):
    fact_layers = []
    ranks = []
    # param_count = 0
    fact_count = 0
    for layer in layers:
        # param_count += count_parameters(layers[0][2])
        fact_layer, rank, error = find_rank_given_error(layer[2], desired_error = desired_error, *args, **kwargs)
        ranks.append(rank)
        fact_layers.append(fact_layer)
        fact_count += count_parameters(fact_layer)
    # compression ratio of all layers
    c = fact_count/baseline_count
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
    return ranks, c, achieved_error
