import os
import json
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

    return paths


def paths_to_df(paths):
    results = []
    for path in paths:
        with open(path / 'results_approximation_error.json') as json_file:
            result = json.load(json_file)
        # print(result)
        for k,v in result.items():
            result[k] = v[0] if type(v) == list else v
        results.append(result)
    
    return pd.DataFrame(results)
