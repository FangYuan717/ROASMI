import os
import sys
import csv
import time
import math
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from logging import Logger
from tqdm import trange,tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
from typing import Callable, Dict, List, Tuple

from args import TrainArgs, PredictArgs
from models import MoleculeModel
from features import set_extra_atom_fdim
from data import MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint, set_cache_graph
from utils import get_data_for_rank,split_data_for_rank,build_lr_scheduler, load_checkpoint, makedirs,\
save_checkpoint, compute_gnorm, compute_pnorm, NoamLR,create_logger, timeit,load_frzn_model,\
param_count_all,param_count, filter_invalid_smiles

def save_preds(pred_1,pred_2,data_1,data_2,pair_probability,path):
    
    pred_1 = [num for elem in pred_1 for num in elem] 
    pred_2 = [num for elem in pred_2 for num in elem] 
    pair_probability = [num for elem in pair_probability for num in elem]
    save_preds_dataframe = pd.DataFrame(data={'smiles1': data_1.smiles(),                                         
                                              'score1': pred_1,
                                              'smiles2':data_2.smiles(),
                                              'score2': pred_2,
                                             'probability': pair_probability})
    save_preds_dataframe.to_csv(path, index=False)
    
def evaluate_accuracy(targets,pair_pr):
    assert len(targets) == len(pair_pr)
    
    score = 0 
    for i in range(len(pair_pr)):
        if abs(pair_pr[i][0] - targets[i][0]) < 0.5:
            score += 1
    accuracy = score/len(pair_pr)
    
    return accuracy

def evaluate_predictions_for_rank(preds,RTs,pair_pr,targets) -> Dict[str, List[float]]:

    results = defaultdict(list)
    results['acc'].append(evaluate_accuracy(targets,pair_pr))
    results['spear'].append(st.spearmanr (preds, RTs)[0])
    results = dict(results)
    
    return results

def predict_for_rank(model: MoleculeModel,data_loader_1: MoleculeDataLoader,data_loader_2: MoleculeDataLoader):

    model.eval()

    pair_pr = []
    targets = []
    output_1=[]
    output_2=[]
    RT_batch = []

    for (batch_1,batch_2) in tqdm(zip(data_loader_1,data_loader_2),total=len(data_loader_1), leave=False):
        mol_batch_1, features_batch_1, target_batch_1 = batch_1.batch_graph(), batch_1.features(), batch_1.targets()
        mol_batch_2, features_batch_2, target_batch_2 = batch_2.batch_graph(), batch_2.features(), batch_2.targets()
        RT_batch.extend(target_batch_1)
        batch_targets = torch.Tensor([[1 if x1<x2 else 0] for (x1,x2) in zip(target_batch_1,target_batch_2)])
        targets.extend(batch_targets)
        
        # Make predictions
        with torch.no_grad():
            output1 = model(mol_batch_1, features_batch_1)
            output2 = model(mol_batch_2, features_batch_2)
        batch_probability = torch.sigmoid(output2-output1) 
        output1=output1.data.cpu().numpy()
        output2=output2.data.cpu().numpy()
        output_1.extend(output1)
        output_2.extend(output2) 
        batch_probability = batch_probability.data.cpu().numpy()
        
        # Collect vectors
        batch_probability = batch_probability.tolist()
        pair_pr.extend(batch_probability)
        
    return output_1,output_2,pair_pr,targets,RT_batch

def evaluate_for_rank(model,data_loader_1,data_loader_2) -> Dict[str, List[float]]:
        
    output_1,output_2,pair_pr,targets,RT_batch = predict_for_rank(model, data_loader_1, data_loader_2)
    results = evaluate_predictions_for_rank( preds=output_1, RTs=RT_batch, pair_pr=pair_pr, targets=targets)
    
    return results

def train_for_rank(model: MoleculeModel,
                   data_loader_1: MoleculeDataLoader,
                   data_loader_2: MoleculeDataLoader,
                   optimizer: Optimizer,
                   scheduler: _LRScheduler,
                   args: TrainArgs,
                   loss_func: Callable,
                   n_iter: int = 0,
                   logger: logging.Logger = None,
                   writer: SummaryWriter = None) -> int:

    debug = logger.debug if logger is not None else print
    
    model.train()
    loss_sum = iter_count = 0    
    
    target =[]
    pair_pr = []
    
    for (batch_1,batch_2) in tqdm(zip(data_loader_1,data_loader_2),total=len(data_loader_1), leave=False):
        # Prepare batch
        batch:MoleculeDataset
        mol_batch_1, features_batch_1, target_batch_1 = batch_1.batch_graph(), batch_1.features(), batch_1.targets()
        mol_batch_2, features_batch_2, target_batch_2 = batch_2.batch_graph(), batch_2.features(), batch_2.targets()
        targets = torch.Tensor([[1 if x1<x2 else 0] for (x1,x2) in zip(target_batch_1,target_batch_2)])
        target.extend(targets)
        
        # Run model
        model.zero_grad()
        output1 = model(mol_batch_1, features_batch_1)
        output2 = model(mol_batch_2, features_batch_2)
        score = torch.sigmoid(output2-output1) 
        targets = targets.to(args.device)
        
        loss = loss_func(score, targets)
        loss_sum += loss.item()
        iter_count += 1
        
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        score = score.data.cpu().numpy().tolist()
        pair_pr.extend(score)

        n_iter += len(batch_1)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')
            
            train_acc = evaluate_accuracy(target,pair_pr)

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)
                writer.add_scalar('train_acc',train_acc,n_iter)

    return n_iter


def run_training_for_rank(args: TrainArgs,data,logger: Logger = None):

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
        
    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Generating paired training batches
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        tr_data,val_data, _= split_data_for_rank(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, num_folds=args.num_folds)          
        tr_data_1,tr_data_2 = get_data_for_rank(data=tr_data, args=args,logger=logger)
        val_data_1,val_data_2 = get_data_for_rank(data=val_data, args=args,logger=logger)
        test_data = pd.read_csv(args.separate_test_path)
        test_data_1, test_data_2 = get_data_for_rank(data=test_data, args=args,logger=logger)
    else:
        tr_data,val_data, test_data = split_data_for_rank(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, num_folds=args.num_folds)          
        tr_data_1,tr_data_2 = get_data_for_rank(data=tr_data, args=args,logger=logger)
        val_data_1,val_data_2 = get_data_for_rank(data=val_data, args=args,logger=logger)
        test_data_1,test_data_2 = get_data_for_rank(data=test_data, args=args,logger=logger)  

    args.features_size = tr_data_1.features_size()
        
    args.train_data_size = len(tr_data_1)
    data_size = len(tr_data_1)+len(val_data_1)+len(test_data_1)
    debug(f'Total size = {data_size:,} | train size = {len(tr_data_1):,} | val size = {len(val_data_1):,} | test size = {len(test_data_1):,}')
    tr_loader_1 = MoleculeDataLoader(dataset=tr_data_1,batch_size=args.batch_size)
    tr_loader_2 = MoleculeDataLoader(dataset=tr_data_2,batch_size=args.batch_size)
    val_loader_1 = MoleculeDataLoader(dataset=val_data_1,batch_size=args.batch_size)
    val_loader_2 = MoleculeDataLoader(dataset=val_data_2,batch_size=args.batch_size)   
    test_loader_1 = MoleculeDataLoader(dataset=test_data_1,batch_size=args.batch_size)
    test_loader_2 = MoleculeDataLoader(dataset=test_data_2,batch_size=args.batch_size)
    
    test_RTs = test_data_1.targets()
    sum_test_preds_1 = np.zeros((len(test_RTs), 1))
    sum_test_preds_2 = np.zeros((len(test_RTs), 1))
    sum_test_pair_pr = np.zeros((len(test_RTs), 1))
    
    loss_func = nn.BCELoss()
    set_cache_graph(False)   
    sum_test_acc = 0
    
    # Train ensemble of ROASMIs
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build ROASMI which is the combination of the D-MPNN embedding module and the RankNet ranking module.
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)
            
        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(model=model,path=args.checkpoint_frzn, current_args=args, logger=logger)     
        
        debug(model)
        
        if args.checkpoint_frzn is not None:
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')
        
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)
        
        # Ensure that ROASMI is saved in correct location for evaluation if 0 epoch
        save_checkpoint(os.path.join(save_dir, args.MODEL_FILE_NAME), model,args)
        
        # Optimizers
        params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]
        optimizer = Adam(params)
        
        start = time.time()
        
        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run (re)training
        best_acc,best_epoch, n_iter = 0, 0, 0
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')
            n_iter = train_for_rank(model,tr_loader_1,tr_loader_2,optimizer,scheduler,args,loss_func,n_iter,logger,writer)
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate_for_rank(model,val_loader_1,val_loader_2)
            for metric, scores in val_scores.items():
                # Average validation rank score
                avg_val_score = np.nanmean(scores)
                debug(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)
                if args.show_individual_scores:
                    # Individual validation rank scores
                    for val_score in scores:
                        debug(f'Validation {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{metric}', val_score, n_iter)

            # Save model checkpoint if improved validation rank score
            avg_val_score = np.nanmean(val_scores['acc'])
            if avg_val_score > best_acc:
                best_acc, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, args.MODEL_FILE_NAME), model, args)
    
        end = time.time()
        train_time = end - start
        if args.checkpoint_frzn is not None:
            debug('The model {} transfer learning train procedure run {:.0f}h {:.0f}m {:.0f}s'.format(model_idx,train_time //3600, (train_time%3600)// 60, train_time % 60)) 
        else:
            debug('The model {} pretrain procedure run {:.0f}h {:.0f}m {:.0f}s'.format(model_idx,train_time //3600, (train_time%3600)// 60, train_time % 60)) 

        # Evaluate on test set using model with best validation rank score
        info(f'The Model {model_idx} best validation accuracy = {best_acc:.4f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, args.MODEL_FILE_NAME), device=args.device, logger=logger)
        if args.cuda:
            model = model.to(args.device) 
        test_1,test_2,test_preds, test_targets,test_RT = predict_for_rank(model,test_loader_1,test_loader_2)

        test_scores = evaluate_predictions_for_rank(test_1,test_RT,test_preds,test_targets) 
        sum_test_preds_1 += np.array(test_1)
        sum_test_preds_2 += np.array(test_2)
        sum_test_pair_pr += np.array(test_preds)
        
        # Average test rank score
        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)
            if args.show_individual_scores:
                # Individual test rank scores
                for test_score in scores:
                    info(f'Model {model_idx} test{metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{metric}', test_score, n_iter)
        writer.close()
        
    # Evaluate ensemble on test set
    avg_test_preds_1 = (sum_test_preds_1 / args.ensemble_size).tolist()
    avg_test_preds_2 = (sum_test_preds_2 / args.ensemble_size).tolist()
    avg_test_pair_pr = (sum_test_pair_pr / args.ensemble_size).tolist()
    ensemble_scores = evaluate_predictions_for_rank(preds=avg_test_preds_1,RTs=test_RTs,pair_pr=avg_test_pair_pr,targets=test_targets)
    for metric, scores in ensemble_scores.items():
        # Average ensemble rank score
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble rank scores
        if args.show_individual_scores:
            for ensemble_score in  scores:
                info(f'Ensemble test {metric} = {ensemble_score:.6f}')
                
    #save test predictions
    path = os.path.join(args.save_dir,args.SAVE_PATH_NAME)
    save_preds(avg_test_preds_1,avg_test_preds_2,test_data_1,test_data_2,avg_test_pair_pr,path)
        
    return ensemble_scores


def cross_validate(args:TrainArgs):
    """Runs k-fold cross-validation.For each of k splits (folds) of the data, trains and tests a model on that split and aggregates the performance across folds."""                 
    logger = create_logger(name=args.TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    debug, info = logger.debug, logger.info

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    makedirs(args.save_dir)

    data = pd.read_csv(args.data_path)

    # Find invalid SMILES,and delete or replace with valid SMILES manually
    filter_invalid_smiles(data)

    # Run training on different random seeds for each fold
    all_scores = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training_for_rank(args, data, logger)

        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
    all_scores = dict(all_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')
    # Report scores for each fold
    for fold_num in range(args.num_folds):
        for metric, scores in all_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')
            if args.show_individual_scores:
                for score in scores[fold_num]:
                    info(f'\t\tSeed {init_seed + fold_num} ==> test retention time {metric} = {score:.6f}')

    # Report scores across folds
    for metric, scores in all_scores.items():
        avg_scores = np.nanmean(scores, axis=1)  
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')
        if args.show_individual_scores:
            info(f'\tOverall test retention time {metric} = {np.nanmean(scores[:, 0]):.6f} +/- {np.nanstd(scores[:, 0]):.6f}')

    # Save scores
    with open(os.path.join(save_dir, args.TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)
        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)
        row = ['RT']
        for metric, scores in all_scores.items():
            task_scores = scores[:, 0]
            mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
            row += [mean, std] + task_scores.tolist()
        writer.writerow(row)

    # Determine mean and std score of main metric
    avg_scores = np.nanmean(all_scores[args.metric], axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

    # Save test split predictions during training
    all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', args.SAVE_PATH_NAME)) for fold_num in range(args.num_folds)])
    all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score

def ROASMI_train():
    cross_validate(args = TrainArgs().parse_args())


def prepare_data_for_prediction(args: PredictArgs = None,logger: Logger = None):

    features_generator = args.features_generator
    
    data = pd.read_csv(args.pred_path)
    filter_invalid_smiles(data)
    data = MoleculeDataset([MoleculeDatapoint(smiles=[smiles],features_generator=features_generator)
                            for i, smiles in enumerate(data['SMILES'])])
    smiles = data.smiles()
    loader = MoleculeDataLoader(dataset=data,batch_size=args.batch_size)

    return smiles, loader


def make_predictions(model: MoleculeModel,
                     data_loader: MoleculeDataLoader,
                     disable_progress_bar: bool = False) -> List[List[float]]:
    model.eval()

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds

def predict_in_real_world(args:PredictArgs):
    
    predict_smiles, predict_loader = prepare_data_for_prediction(args)
    sum_preds = []
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        model = load_checkpoint(checkpoint_path, device=args.device)
        if args.cuda:
            model = model.to(args.device)        
        model_preds = make_predictions(model=model, data_loader=predict_loader) 
        sum_preds.append(np.array(model_preds))

    # Ensemble predictions
    sum_preds = sum(sum_preds)
    avg_preds = sum_preds / len(args.checkpoint_paths)
    #avg_preds = avg_preds.tolist()

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    #assert len(predict_data) == len(avg_preds)
    path = os.path.join(args.preds_path,args.SAVE_NAME)
    makedirs(path, isfile=True)
    avg_preds = [num for elem in avg_preds for num in elem] 
    save_preds_dataframe = pd.DataFrame(data={'smiles': predict_smiles, 'score': avg_preds})
    save_preds_dataframe.to_csv(path, index=False)

    return avg_preds

def ROASMI_predict():
    predict_in_real_world(args = PredictArgs().parse_args())
