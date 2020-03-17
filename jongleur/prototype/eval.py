import importlib.util as imp
import json

import click


@click.command()
@click.option('--nn', type=str, default="./model.py")
@click.option('--nn_params', type=str, default="{}")
@click.option('--dataset', type=str)
def sum(nn, nn_params, dataset):
    spec = imp.spec_from_file_location("my_module", nn)
    module = imp.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_params = json.loads(nn_params)
    model = module.Model(**model_params)
    print(model.eval())

    model.summarize(dataset)


if __name__ == '__main__':
    sum()
