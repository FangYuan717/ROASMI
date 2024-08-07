This repository contains data and codes allowing researchers to recreate the entire analysis for the manuscript:

"ROASMI: Accelerating Small Molecule Identification by Repurposing Retention Data",

ROASMI is a Retention Order model to Assist Small Molecule Identification. 
We provide four initial ROASMI models () and two re-trained ROASMI models(). 
The former applies to experiments using an RPLC system with an eluent pH of around 2.7, while the latter applies to HILIC systems with a similar acidic eluent pH level.
The user can predict the retention score based on SMILES of candidate analytes.

To use the ROASMI to assist 

git clone https://github.com/chemprop/chemprop.git
cd chemprop
conda env create -f environment.yml
conda activate chemprop
pip install -e .
