# ROASMI
Support for retention prediction in LC-MS-based small molecule identification.  
# Overview  
This repository contains Retention Order-Assisted Small Molecule Identification (ROASMI) for predicting the retention property of compounds across datasets.
## The molecular embedding module
This module is an extension of chemprop described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) which is available in the [chemprop GitHub repository](https://github.com/swansonk14/chemprop). A directed message transfer neural network (D-MPNN) is employed to learn directly from the structure of compounds, thus enabling the predictions of compounds in new chemical spaces: molecules can be mapped directly into generic chemical spaces highly attuned to the desired property without first being mapped into artificially designed chemical spaces of fingerprints or descriptors.	  
## The retention prediction module
A Ranking Neural Network (RankNet) was used to learn elution orders from replicable retention sequences in reference sets with similar pH conditions and same chromatographic systems (such as reverse phase). Learning-to-rank enables the predictions to a wide range of chromatographic spaces: pairwise retention order can be predicted across datasets without first mapping the pointwise retention value to a common gradient range. Additional benefit is learning-to-rank supports simultaneous learning from multiple heterogeneous datasets.	  
# Requirements	
It is possible to run all of the code on a CPU-only machine. We recommend using a GPU for large re-training sets (＞1000 molecules) which can significantly faster re-training.	  
To run ROASMI with GPUs, you will need:	  
  •	cuda >= 8.0	  
  •	cuDNN  
Meantime, you may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/).	
# Installation	
1.	`git clone https://github.com/FangYuan717/ROASMI.git`
2.	`cd /path/to/ROASMI`
3.	`conda env create -f environment.yml`
4.	`conda activate ROASMI`
5.	`pip install -e .`
6.	(Optional) `pip install git+https://github.com/bp-kelley/descriptastorus`    
The optional descriptastorus package is only necessary if you plan to incorporate computed RDKit features into your model. The initial ROASMI we offer does not include these features because they were not shown performance gains during training.
# Retraining
To retrain initial ROASMI, run:  
`python ROASMI/ROASMI_train.py --data_path <path1> --separate_test_path <path2> --checkpoint_paths <path3> --save_dir <dir>`  
where <path1> is the path to reference data CSV files which includes system, SMILES and retention times; <path2> is the path to test data CSV files (optional); <path3> is the path for the initial model checkpoint which is defaulted as “path/to/ROASMI.pt”; and <dir> is the directory where model checkpoints will be saved.  
Detailed retraining configuration can see `TrainArgs` in [args.py](https://github.com/FangYuan717/ROASMI/blob/main/args.py).
# Predicting
To predict retention scores of each compound in given datasets, run:  
`python ROASMI/ROASMI_predict.py --pred_path <path4> --preds_path <path5>`  
where <path4> is the path to ready-to-predict data CSV files which only needs list of SMILES; <path5> is the path to predicted data CSV file.  
Detailed retraining configuration can see `PredictArgs` in [args.py](https://github.com/FangYuan717/ROASMI/blob/main/args.py).  
# Identifying
This module is an extension of probabilistic framework described in the paper [Probabilistic framework for integration of mass spectrum and retention time information in small molecule identification](https://academic.oup.com/bioinformatics/article/37/12/1724/6007259?login=true) which is available in the [GitHub repository](https://github.com/aalto-ics-kepaco/msms_rt_score_integration).   
To annotate small molecules in given datasets, we integrate predicted retention scores with MS/MS scores by running:  
`python ROASMI/Identify.py --margain_type <choice> --n_trees<num1> --D<num2> --identify_path <path6>`  
where <choice> is the choice of marginal: max- or sum-marginal; <num1> is the number of trees used to average the marginal distribution; <num2> is a scalar of weight of the former between the retention scores and MS/MS scores; <path6> is path to ready-to-identify data CSV files, which includes peak ID, retention times, SMILES, MS/MS score, and retention scores.  
Detailed retraining configuration can see `IdentifyArgs` in [args.py](https://github.com/FangYuan717/ROASMI/blob/main/args.py).
