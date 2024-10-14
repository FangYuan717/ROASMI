# ROASMI
ROASMI is a Retention Order model to Assist Small Molecule Identification.   
In addition to code and trained models, this repository also contains data that allows researchers to recreate the entire analysis for the manuscript:  
**“ROASMI: Repurposing retention data to aid small molecule identification”** .  
We provide four initial ROASMI models (ROASMI_1 - ROASMI_5) for predicting the retention behavior of compounds in the reversed-phase liquid chromatography (RPLC) system with an eluent pH of around 2.7.  
The ensemble approach allowed quantifying model uncertainty using the variance of the retention order predictions across the trained models.   
## Overview
Designed to predict retention scores for candidate compounds during small molecule identification, ROASMI consists of two main modules.    
**The molecular embedding module.** This module is an extension of chemprop described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) which is available in the [chemprop GitHub repository](https://github.com/chemprop/chemprop). A directed message transfer neural network (D-MPNN) is used to learn directly from the structure of compounds, allowing prediction of compounds in new chemical spaces: molecules can be mapped directly into generic chemical spaces that are highly attuned to the desired property without first being mapped into artificially designed chemical spaces of fingerprints or descriptors.   
**The retention prediction module.** A Ranking Neural Network (RankNet) was used to learn elution orders from consistent retention sequences in reference sets with similar pH conditions and same chromatographic systems (e.g. reversed phase). Learning-to-rank enables the predictions to be made across a wide range of chromatographic spaces: pairwise retention order can be predicted across datasets without first mapping the pointwise retention value to a common gradient range. As an additional benefit, learning-to-rank supports simultaneous learning from multiple heterogeneous datasets.    
## Requirements
All code can be run on a CPU-only machine; however, for faster learning speed, we recommend using GPUs for retraining.    
To run ROASMI with GPUs, you need:   
  •	cuda >= 8.0  
  •	cuDNN  
Meantime, you may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/).  
## Installation  
  1.	git clone https://github.com/FangYuan717/ROASMI.git
  2.	cd /path/to/ROASMI
  3.	conda env create -f environment.yml
  4.	conda activate ROASMI
  5.	pip install -e .
  6.	(Optional) pip install git+https://github.com/bp-kelley/descriptastorus  
The optional descriptastorus package is only necessary if you plan to incorporate computed RDKit features into your model. The ROASMI models we provide do not include these features as they have not shown performance gains during training.      
## Retraining (optional)
To retrain ROASMI, run:  
  `python code/ROASMI_train.py`  
See `TrainArgs` in args.py for parameter changes. The default retraining is to retrain two HILIC models using 'dataset_83' as the reference set and 'dataset_84' as the test set.  
## Predicting
To predict the retention scores of each compound in the ready-to-identify datasets, run:  
  `python code/ROASMI_predict.py`   
See`PredictArgs` in args.py for parameter changes. The default prediction is made using the ROASMI_2 and the default pred_path is 'data/predict_toy_set. csv'.  
## Identifying
This module is an extension of probabilistic framework described in the paper [Probabilistic framework for integration of mass spectrum and retention time information in small molecule identification](https://academic.oup.com/bioinformatics/article/37/12/1724/6007259?login=true) which is available in the [GitHub repository](https://github.com/aalto-ics-kepaco/msms_rt_score_integration).   
To annotate small molecules in ready-to-identify datasets, we integrate predicted retention scores with MS/MS scores by running:  
  `python code/Identify.py`  
See `IdentifyArgs` in args.py for parameter changes. User-supplied files to be identified can be referenced as 'identify_toy_set.csv' in the 'data' folder .
# License
This project is licensed under the MIT License.
# External Archive
This project is also archived on Zenodo with the following DOI: [10.5281/zenodo.13927187](https://doi.org/10.5281/zenodo.13927187)
