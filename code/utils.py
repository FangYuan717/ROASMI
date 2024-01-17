import os
import re
import csv
import math
import pickle
import logging
import collections
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from rdkit import Chem
from random import Random
from logging import Logger
from functools import wraps
from argparse import Namespace
from datetime import timedelta
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from data import MoleculeDataLoader, MoleculeDataset,MoleculeDataset, MoleculeDatapoint
from args import TrainArgs, PredictArgs 
from models import MoleculeModel


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str,model: MoleculeModel,args: TrainArgs = None) -> None:
    """
    Saves a model checkpoint.
    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    state = {
        'args': args,
        'state_dict': model.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(path: str,device: torch.device = None,logger: logging.Logger = None) -> MoleculeModel:

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    #Loads arguments from a dictionary, ensuring all required arguments are set
    args.from_dict(vars(state['args']), skip_unsettable=True)
    loaded_state_dict = state['state_dict']

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
            param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(f'Warning: Pretrained parameter "{loaded_param_name}" '
                 f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug('Moving model to cuda')
    model = model.to(args.device)

    return model

def overwrite_state_dict(loaded_param_name: str,
                        model_param_name: str,
                        loaded_state_dict: collections.OrderedDict,
                        model_state_dict: collections.OrderedDict,
                        logger: logging.Logger = None) -> collections.OrderedDict:

    debug = logger.debug if logger is not None else print

    
    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')
        
    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(f'Pretrained parameter "{loaded_param_name}" '
              f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
              f'model parameter of shape {model_state_dict[model_param_name].shape}.')
    
    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]    
    
    return model_state_dict


def load_frzn_model(model: torch.nn,
                    path: str,
                    current_args: Namespace = None,
                    logger: logging.Logger = None) -> MoleculeModel:

    debug = logger.debug if logger is not None else print

    loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model['state_dict']
    loaded_args = loaded_mpnn_model['args']

    model_state_dict = model.state_dict()
       
    encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']
    if current_args.checkpoint_frzn is not None:
        # Freeze the MPNN
        for param in model.encoder.parameters():
            param.requires_grad = False 
        for param_name in encoder_param_names:
            model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)
        
    if current_args.frzn_ffn_layers > 0:         
        ffn_param_names = [['ffn.'+str(i*3+1)+'.weight','ffn.'+str(i*3+1)+'.bias'] for i in range(current_args.frzn_ffn_layers)]
        ffn_param_names = [item for sublist in ffn_param_names for item in sublist]     
        # Freeze MPNN and FFN layers
        for name,param in model.ffn.named_parameters():
            if "1." in name:
                param.requires_grad = False
        for param_name in encoder_param_names+ffn_param_names:
            model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)               
    
    # Load pretrained weights
    model.load_state_dict(model_state_dict)
    
    return model


def load_args(path: str) -> TrainArgs:

    args = TrainArgs()
    args.from_dict(vars(torch.load(path, map_location=lambda storage, loc: storage)['args']), skip_unsettable=True)

    return args
    
def build_lr_scheduler(optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None) -> _LRScheduler:

    return NoamLR(optimizer=optimizer,warmup_epochs=[args.warmup_epochs], total_epochs=total_epochs or [args.epochs],
                  steps_per_epoch=args.train_data_size // args.batch_size, init_lr=[args.init_lr], max_lr=[args.max_lr],final_lr=[args.final_lr])


def create_logger(name: str, save_dir: str, quiet: bool) -> logging.Logger:

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:

    def timeit_decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f'Elapsed time = {delta}')

            return result

        return wrap

    return timeit_decorator

def save_features(path: str, features: List[np.ndarray]) -> None:

    np.savez_compressed(path, features=features)

def compute_pnorm(model: nn.Module) -> float:

    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:

    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def param_count(model: nn.Module) -> int:

    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def param_count_all(model: nn.Module) -> int:

    return sum(param.numel() for param in model.parameters())


def compute_molecule_vectors(model: nn.Module,
                             data: MoleculeDataset,
                             batch_size: int,
                             num_workers: int = 8) -> List[np.ndarray]:

    training = model.training
    model.eval()
    data_loader = MoleculeDataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    vecs = []
    for batch in tqdm(data_loader, total=len(data_loader)):
        # Apply model to batch
        with torch.no_grad():
            batch_vecs = model.featurize(batch.batch_graph(), batch.features())

        # Collect vectors
        vecs.extend(batch_vecs.data.cpu().numpy())

    if training:
        model.train()

    return vecs


class NoamLR(_LRScheduler):

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:

        return list(self.lr)

    def step(self, current_step: int = None):

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

def pick_pair_for_rank(group_one_data, args: Union[TrainArgs, PredictArgs]):
    
    RT_interval = args.RT_interval 
    j=0
    pair_data=pd.DataFrame(columns=('System','SMILES','RT','SMILES1','RT1'))
    group_one_data_num=len(group_one_data)
    for i in range(group_one_data_num-1):
        while j<group_one_data_num-1:
            j=j+1                        
            if(group_one_data.iloc[j,2]-group_one_data.iloc[i,2]>=RT_interval):
                pair_data=pair_data.append(pd.DataFrame({'SMILES':[group_one_data.iloc[i,1]],'RT':[group_one_data.iloc[i,2]],
                                                         'System':[group_one_data.iloc[i,0]],'SMILES1':[group_one_data.iloc[j,1]],
                                                         'RT1':[group_one_data.iloc[j,2]]}),ignore_index=True)
                break
    return pair_data

def filter_invalid_smiles(data):
    
    invalid_SMILES = []
    for datapoint in tqdm(data['SMILES']):
        mol = Chem.MolFromSmiles(datapoint)
        if mol is None or mol.GetNumHeavyAtoms()<= 0:
            invalid_SMILES.append(datapoint)
    print(f'Warning: {len(invalid_SMILES)} SMILES are invalid:{invalid_SMILES}.')
    
def split_data_for_rank(data, split_type: str,sizes: Tuple[float, float, float],seed: int, num_folds: int):

    random = Random(seed)
    if split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        tr_data = pd.DataFrame(columns = ('System','SMILES','RT'))
        val_data = pd.DataFrame(columns = ('System','SMILES','RT'))
        test_data = pd.DataFrame(columns = ('System','SMILES','RT'))
        for i in indices[:train_size]:
            tr_data = tr_data.append(pd.DataFrame({'System':[data.values[i,0]],'SMILES':[data.values[i,1]],'RT':[data.values[i,2]]}),ignore_index=True) 
        for i in indices[train_size:train_val_size]:
            val_data = val_data.append(pd.DataFrame({'System':[data.values[i,0]],'SMILES':[data.values[i,1]],'RT':[data.values[i,2]]}),ignore_index=True) 
        for i in indices[train_val_size:]:
            test_data = test_data.append(pd.DataFrame({'System':[data.values[i,0]],'SMILES':[data.values[i,1]],'RT':[data.values[i,2]]}),ignore_index=True)
    
    elif split_type == 'cv':
        random = Random(0)
        
        num_folds = TrainArgs.num_folds
        indices = np.repeat(np.arange(num_folds), 1 + len(data) // num_folds)[:len(data)]
        random.shuffle(indices)
        test_index = seed % num_folds
        val_index = (seed + 1) % num_folds

        tr_data = pd.DataFrame(columns=('System','SMILES','RT'))
        val_data = pd.DataFrame(columns=('System','SMILES','RT'))
        test_data = pd.DataFrame(columns=('System','SMILES','RT'))

        for d, index in zip(data.values, indices):
            if index == test_index:
                test_data = test_data.append(pd.DataFrame({'System':[d[0]],'SMILES':[d[1]],'RT':[d[2]]}),ignore_index=True) 
            elif index == val_index:
                val_data = val_data.append(pd.DataFrame({'System':[d[0]],'SMILES':[d[1]],'RT':[d[2]]}),ignore_index=True) 
            else:
                tr_data = tr_data.append(pd.DataFrame({'System':[d[0]],'SMILES':[d[1]],'RT':[d[2]]}),ignore_index=True)  
      
    return tr_data,val_data,test_data

def get_data_for_rank(data, args: Union[TrainArgs, PredictArgs],logger: Logger)-> MoleculeDataset:
    
    features_generator = args.features_generator
    data_system_group = np.unique(data.iloc[:,0])                             
    system_group_num = len(data_system_group)
    pair_data_all = pd.DataFrame(columns=('System','SMILES','RT','SMILES1','RT1'))
    for k in range(system_group_num):
        system_group_one_data = data[data["System"]==data_system_group[k]]
        system_group_one_data = system_group_one_data.sort_values(by="RT")
        system_group_one_data.reset_index(drop=True)
        #group_one_data=system_group_one_data
        pair_data_one = pick_pair_for_rank(system_group_one_data,args)
        pair_data_all = pair_data_all.append(pair_data_one)   

    data1 = MoleculeDataset([MoleculeDatapoint(smiles=[smiles],targets=targets,features_generator=features_generator)
                             for i, (smiles, targets) in enumerate(zip(pair_data_all['SMILES'], pair_data_all['RT']))])
    data2 = MoleculeDataset([MoleculeDatapoint(smiles=[smiles],targets=targets,features_generator=features_generator)
                             for i, (smiles, targets) in enumerate(zip(pair_data_all['SMILES1'], pair_data_all['RT1']))])
    
    return data1,data2
