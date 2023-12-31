"""Optimizes hyperparameters using Bayesian optimization."""

import os
import json
import numpy as np

from copy import deepcopy
from typing import Dict, Union
from hyperopt import fmin, hp, tpe

from args import HyperoptArgs
from models import MoleculeModel 
from ROASMI import cross_validate
from utils import create_logger, makedirs, timeit, param_count


SPACE = {
    'hidden_size': hp.quniform('hidden_size', low = 1000, high = 1500, q = 100),
    'dropout': hp.quniform('dropout', low = 0.0, high = 0.3, q = 0.1),
    'ff1_hidden_size': hp.quniform('ff1_hidden_size', low = 50, high = 700, q = 100)
}
INT_KEYS = ['hidden_size', 'ff1_hidden_size']

def hyperopt(args: HyperoptArgs) -> None:
    """
    Runs hyperparameter optimization on a Chemprop model.
    Hyperparameter optimization optimizes the following parameters:
    * :code:`hidden_size`: The hidden size of the neural network layers is selected from {300, 400, ..., 2400}
    * :code:`depth`: The number of message passing iterations is selected from {2, 3, 4, 5, 6}
    * :code:`dropout`: The dropout probability is selected from {0.0, 0.05, ..., 0.4}
    The best set of hyperparameters is saved as a JSON file to :code:`args.config_save_path`.
    :param args: A :class:`~chemprop.args.HyperoptArgs` object containing arguments for hyperparameter
                 optimization in addition to all arguments needed for training.
    """
    # Create logger
    logger = create_logger(name=args.HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Copy args
        hyper_args = deepcopy(args)

        # Update args with hyperparams
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)

        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)        

        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        mean_score, std_score = cross_validate(hyper_args)

        # Record results
        temp_model = MoleculeModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })
        return - 1 * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters, rstate=np.random.RandomState(args.seed))

    # Report best result
    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = max(results, key=lambda result: result['mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def RORC_hyperopt() -> None:
    """Runs hyperparameter optimization for a Chemprop model.
    This is the entry point for the command line command :code:`chemprop_hyperopt`.
    """
    hyperopt(args=HyperoptArgs().parse_args())


if __name__ == '__main__':
    RORC_hyperopt()
