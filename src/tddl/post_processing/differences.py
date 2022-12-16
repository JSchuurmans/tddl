import torch
from tensorly import tensor_to_vec
from numpy.linalg import norm
from pandas import DataFrame
# from torch.linalg import norm

from tltorch.factorized_tensors import TuckerTensor, CPTensor, TTTensor

def weight_difference(weights_a, weights_b, layer_nrs, fact_nr, contract=True):

    df = DataFrame(columns=['layer_nr', 'fact_nr', 'norm_2', 'norm_1', 'diff','relative_norm_2','relative_norm_1'])
    for nr in layer_nrs:
        with torch.no_grad():
            
            if contract and type(weights_a[nr]) in [TuckerTensor, CPTensor, TTTensor]:
                print(f'contracted tensor of type: {type(weights_a[nr])}')
                a = tensor_to_vec(weights_a[nr].to_tensor())
                b = tensor_to_vec(weights_b[nr].to_tensor())
            else:
                a = tensor_to_vec(weights_a[nr])
                b = tensor_to_vec(weights_b[nr])

            diff = (a - b).cpu().numpy()
        
            # norm on difference
            norm_2 = norm(diff)
            norm_1 = norm(diff, ord=1)
            
            relative_norm_2 = norm_2 / norm(a.cpu().numpy())
            relative_norm_1 = norm_1 / norm(a.cpu().numpy(), ord=1)
            
        df = df.append({
            'layer_nr':nr,
            'fact_nr':fact_nr,
            'norm_2':norm_2,
            'norm_1':norm_1,
            'diff': diff,
            'relative_norm_2': relative_norm_2,
            'relative_norm_1': relative_norm_1,
        }, ignore_index=True)

    return df
