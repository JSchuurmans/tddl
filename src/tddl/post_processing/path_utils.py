import os
import json
import yaml
from yaml import Loader
from pathlib import Path

import pandas as pd


def logdir_to_paths(logdir: Path):

    paths = []
    folders = os.listdir(logdir)
    for folder in folders:
        for subfolder in os.listdir(logdir / folder):
            if 'lr' in subfolder:
                if (logdir / folder / subfolder / "results.json").exists():
                    paths.append( logdir / folder / subfolder )
                else:
                    print(f'"results.json" does not exist in {logdir / folder / subfolder}')
    return paths


def paths_to_df(paths, filename = 'results_approximation_error.json', dbs=False):
    results = []
    for path in paths:
        with open(path / filename) as json_file:
            result = json.load(json_file)
        # print(result)
        if not dbs:
            for k,v in result.items():
                result[k] = v[0] if type(v) == list else v
        results.append(result)
    
    return pd.DataFrame(results)


def logdir_to_df(logdir: Path):

    df = pd.DataFrame()
    folders = os.listdir(logdir)
    for folder in folders:
        config_path = logdir / folder / "config.yml"

        with open(config_path) as yaml_file:
            config = yaml.load(yaml_file, Loader=Loader)

        for subfolder in os.listdir(logdir / folder):
            if 'lr' in subfolder:
                logdir_model = logdir / folder / subfolder
                config['model_best'] = logdir_model / "fact_model_best.pth"
                config['model_final'] = logdir_model / "fact_model_final.pth"
                config['model_fact'] = logdir_model / "model_after_fact.pth"

        layers = config['layers']
        if len(layers) == 1:
            config['layer'] = int(config['layers'][0])

        df = df.append(config, ignore_index=True)

        df['rank'] = df['rank'].astype(float)

    return df
