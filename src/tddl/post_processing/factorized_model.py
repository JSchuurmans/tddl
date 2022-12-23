import yaml
import json
from pathlib import Path

import torch
import typer

from tddl.factorizations import number_layers
from tddl.factorizations import listify_numbered_layers
from tddl.utils.approximation import calculate_relative_error, calculate_scaled_error, calculate_error
from tddl.post_processing.path_utils import logdir_to_paths


def process_factorized_networks(paths, baseline_path):

    baseline_model = torch.load(baseline_path / "cnn_best.pth")
    with open(baseline_path/'results.json') as json_file:
        baseline_result = json.load(json_file)

    for path in paths:
        # print(path)
        path = Path(path)
        config_path = path.parents[0] / "config.yml"
        config_data = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        layers = config_data['layers']

        fact_model = torch.load(path / 'model_after_fact.pth')

        pretrained_numbered_layers = number_layers(baseline_model)
        pretrained_layers = listify_numbered_layers(
            pretrained_numbered_layers,
            layer_nrs=layers,
        )

        decomposed_numbered_layers = number_layers(fact_model)
        decomposed_conv_layers = listify_numbered_layers(
            decomposed_numbered_layers,
            layer_nrs=layers,
        )

        with open(path / 'results.json') as json_file:
            result = json.load(json_file)

        with open(path / 'results_before_training.json') as json_file:
            result_before_training = json.load(json_file)

        errors_conv = {
                'name': [],
                'nr': [],
                'relative_norm_weight': [],
                'scaled_norm_weight': [],
                'diff_norm_weight': [],
                'layers': layers,
                'factorization': config_data['factorization'],
                'rank': config_data['rank'],
                'valid_acc': result['best_valid_acc'],
                'valid_acc_before_ft': result_before_training['valid_acc'],
                'test_acc_before_ft': result_before_training['test_acc'],
                'n_param_fact': result['n_param_fact'],
                'test_acc': result['test_acc'],
                'lr': config_data['lr'],
                'optimizer': config_data['optimizer'],
            }

        with open(path / 'results_feature_metrics_train.json') as json_file:
            feature_result = json.load(json_file)
        train_features = feature_result['mean']
        errors_conv.update(train_features)

        # with open(path / 'results_feature_metrics_valid.json') as json_file:
        #     feature_result = json.load(json_file)
        # errors_conv.update()

        for pre, dec in zip(pretrained_layers, decomposed_conv_layers):
            
            if pre[0] != dec[0]:
                print(f'breaking: {pre[0]} != {dec[0]}')
                break
            if pre[1] != dec[1]:
                print(f'breaking: {pre[1]} != {dec[1]}')
                break
            
            name = pre[0]
            # print(name)
            nr = pre[1]
            # print(nr)

            pre_weight = pre[2].weight
            dec_weight = dec[2].weight.to_tensor()
            if config_data['factorization'] == 'tt':
                dec_weight = dec_weight.permute(3, 0, 1, 2)

            relative_norm_weight = calculate_relative_error(pre_weight, dec_weight)
            scaled_norm_weight = calculate_scaled_error(pre_weight, dec_weight)
            diff_norm_weight = calculate_error(pre_weight, dec_weight)
            
            errors_conv['name'].append(name)
            errors_conv['nr'].append(nr)
            errors_conv['relative_norm_weight'].append(float(relative_norm_weight))
            errors_conv['scaled_norm_weight'].append(float(scaled_norm_weight))
            errors_conv['diff_norm_weight'].append(float(diff_norm_weight))

        errors_path = path / 'results_approximation_error.json'
        with open(errors_path, 'w') as f:
            json.dump(errors_conv, f)


def main(
    logdir = Path("/bigdata/cifar10/logs/garipov/decomposed/"),
    baseline_path = Path("/bigdata/cifar10/logs/garipov/baselines/1647358615/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue"),
):
    paths = logdir_to_paths(logdir)
    process_factorized_networks(paths, baseline_path)

if __name__ == "__main__":
    typer.run(main)
