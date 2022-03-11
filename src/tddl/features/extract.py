import random
from pathlib import Path
from typing import List
import json
import os

import yaml
from tqdm import tqdm
import numpy as np
import torch
from torch import linalg as LA
import typer
import pandas as pd

from tddl.data.loaders import fetch_loaders
from tddl.utils.random import set_seed


app = typer.Typer()

outputs= []
def hook(module, input, output):
    outputs.append(output)


def hook_network(
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
                    m._modules[name].register_forward_hook(hook)
                try:
                    # if verbose and return_error:
                    #     print((i, error))
                    output[name] = (i, nested_children(child, **kwargs) )
                except TypeError:
                    output[name] = (i, nested_children(child, **kwargs) )
        return output #, errors
    out = nested_children(model, **kwargs)

    if return_error:
        return out


def _get_loaders(
    data_dir: Path,
    dataset: str = 'cifar10',
    batch_size: int = 1,
    data_workers: int = 4
):
    train_loader, valid_loader, test_loader = fetch_loaders(
        dataset=dataset,
        path=data_dir,
        batch_size=batch_size,
        data_workers=data_workers,
        valid_size=5000,
        random_transform_training=False,
    )
    
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
    }


@app.command()
def process_features(
    path: Path,
    split: str = 'train',
    **kwargs,
):
    """
        input:
            dataset: default train [train, valid, test]
    """
    # for path in log_paths:
    # fact_path = path / "fact_model_best.pth"
    fact_path = path / "model_after_fact.pth"

    config_path = path.parent / "config.yml"
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    baseline_path = config['baseline_path']
    layers = config['layers']
    data_dir = Path(config['data_dir'])

    fact_model = torch.load(fact_path)
    hook_network(fact_model, layers=layers)

    baseline_model = torch.load(baseline_path)
    hook_network(baseline_model, layers=layers)
    # TODO: check if outputs needs to be different list / locally defined
    
    loaders = _get_loaders(data_dir, **kwargs)
    loader = loaders.get(split)

    # mkdir output for dataset
    output_dir = path / "features" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(loader)):
            img = img.cuda()

            _ = baseline_model(img)
            baseline_features = outputs.pop()

            _ = fact_model(img)
            fact_features = outputs.pop()

            assert baseline_features.size() == fact_features.size()

            # metrics
            norm_diff = LA.norm(baseline_features - fact_features).item() # 

            norm_b = LA.norm(baseline_features).item()
            n_b = baseline_features.numel()

            metrics = {
                "norm_diff": norm_diff,
                "norm_b": norm_b,
                "n_b": n_b,
            }

            with open(output_dir / f"features_{i}.json", 'w') as outfile:
                json.dump(metrics, outfile)


@app.command()
def aggregate_results(
    path: Path,
    split: str = 'train',
):
    norms = []
    features_path = path / 'features' / split
    for filename in os.listdir(features_path):
        with open(features_path / filename) as json_file:
            data = json.load(json_file)
            norms.append(data)

    df = pd.DataFrame(norms)
    
    df['relative_norm'] = df.norm_diff / df.norm_b
    df['scaled_norm'] = df.norm_diff / df.n_b
    feature_metrics = {
        'mean': dict(df.mean()),
        'std': dict(df.std()),
        'median': dict(df.median()),
    }

    with open(path / f"results_feature_metrics_{split}.json", 'w') as outfile:
        json.dump(feature_metrics, outfile)


@app.command()
def main(
    path: Path, 
    data_workers: int = 8,
    seed: int = 0,
    split: str = 'train',
    dataset: str = 'cifar10',
    aggregate: bool = False,
):

    set_seed(seed)

    # generate list from path
    for nr in tqdm(os.listdir(path)):
        path_nr = path / nr
        dirs = [d for d in os.listdir(path_nr) if os.path.isdir(path_nr / d)]
        process_features(
            path = path_nr / dirs[0],
            split=split,
            data_workers=data_workers,
            dataset=dataset,
        )

        if aggregate:
            aggregate_results(
                path_nr / dirs[0], 
                split,
            )


if __name__ == "__main__":
    app()
