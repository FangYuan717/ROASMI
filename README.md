# ROASMI
Code and initial model for the retention scores prediction in the “ROASMI: Repurposing retention data to aid small molecule identification” manuscript.  
## Overview
**The molecular embedding module.** This module is an extension of chemprop described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) which is available in the [chemprop GitHub repository](https://github.com/chemprop/chemprop). A directed message transfer neural network (D-MPNN) is employed to learn directly from the structure of compounds, thus enabling the predictions of compounds in new chemical spaces: molecules can be mapped directly into generic chemical spaces highly attuned to the desired property without first being mapped into artificially designed chemical spaces of fingerprints or descriptors. 
**The retention prediction module.** A Ranking Neural Network (RankNet) was used to learn elution orders from replicable retention sequences in reference sets with similar pH conditions and same chromatographic systems (such as reverse phase). Learning-to-rank enables the predictions to a wide range of chromatographic spaces: pairwise retention order can be predicted across datasets without first mapping the pointwise retention value to a common gradient range. Additional benefit is learning-to-rank supports simultaneous learning from multiple heterogeneous datasets.  
## Requirements
All code can be run on a CPU-only machine; however, for faster learning speed, we recommend using GPUs for retraining.
To run ROASMI with GPUs, you will need:
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
The optional descriptastorus package is only necessary if you plan to incorporate computed RDKit features into your model. The initial ROASMI we offer does not include these features because they were not shown performance gains during training.  
## Retraining
To retrain ROASMI, run:
  python code/ROASMI_train.py 
Refer to `TrainArgs` in args.py for parameter changes. There are two initial models: ROASMI_1.pt for RPLC acidic conditions and ROASMI_2.pt for HILIC acidic conditions. If the parameters are not modified, the default settings involve retraining ROASMI_2 using 'dataset_83' as the reference set to predict 'dataset_84'.  
## Predicting
To predict retention scores of each compound in given datasets, run:
  python code/ROASMI_predict.py 
Refer to `PredictArgs` in args.py for parameter changes. The default prediction is made using the ROASMI_2 and the default pred_path is 'data/predict_toy_set. csv'.  
## Identifying
This module is an extension of probabilistic framework described in the paper [Probabilistic framework for integration of mass spectrum and retention time information in small molecule identification](https://academic.oup.com/bioinformatics/article/37/12/1724/6007259?login=true) which is available in the [GitHub repository](https://github.com/aalto-ics-kepaco/msms_rt_score_integration). 
To annotate small molecules in given datasets, we integrate predicted retention scores with MS/MS scores by running:
  python code/Identify.py
Refer to `IdentifyArgs` in args.py for parameter changes. User-supplied files to be identified can be referenced in the 'data' folder as 'identify_toy_set.csv'.
