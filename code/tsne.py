
import os
import sys
import csv
import time
import numpy as np
from tap import Tap
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from typing import List
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from features import get_features_generator
from utils import makedirs

class Args(Tap):
    smiles_paths: List[str] =['E:/2022.10/input/SMRT.csv','E:/2022.10/input/Vesna_Vasić.csv','E:/2022.10/input/MTBLS39.csv','E:/2022.10/input/De_Paepe.csv','E:/2022.10/input/Zhenzuo_Jiang.csv','E:/2022.10/input/MTBLS1572_StdMixHigh.csv','E:/2022.10/input/MPI_Symmetry.csv','E:/2022.10/input/Wei_Jia.csv','E:/2022.10/input/CS21.csv','E:/2022.10/input/ChangliangYao.csv','E:/2022.10/input/ST001095_neg.csv','E:/2022.10/input/Musenga.csv','E:/2022.10/input/DynaStI.csv']
    # Path to .csv files containing smiles strings (with header)
    smiles_column: str = None  # Name of the column containing SMILES strings for the first data. By default, uses the first column.
    sizes: List[float] = [3,5,5,5,5,5,5,5,5,5,5,5]  # Sizes of the points associated with each molecule
    scale: int = 1  # Scale of figure
    plot_molecules: bool = False # Whether to plot images of molecules instead of points
    colors: List[str] = ["#8696A7","#AF8C64","#B1735B","#4EAAA0","#5592C1","#B185A4","#AFB0B2","#DFBEAE","#739EA9","#7C8D7A","#E7DDBA","#EBC98A"]  # Colors of the points associated with each dataset ['red', 'green', 'orange', 'purple', 'blue','black']
    max_per_dataset: int = 1500  # Maximum number of molecules per dataset; larger datasets will be subsampled to this size
    save_path: str = 'E:/2022.10/output/#1212'  # Path to a .png file where the t-SNE plot will be saved
    


def compare_datasets_tsne(args: Args):

    # Random seed for random subsampling
    np.random.seed(0)

    # Load the smiles datasets
    print('Loading data')
    smiles, slices, labels = [], [], []
    for smiles_path in args.smiles_paths:
        # Get label
        label = os.path.basename(smiles_path).replace('.csv', '')

        # Get SMILES
        with open(smiles_path) as f:
            reader = csv.DictReader(f)
            new_smiles = [row['SMILES'] for row in reader]
        #new_smiles = get_smiles(path=smiles_path, smiles_columns=args.smiles_column, flatten=True)
        print(f'{label}: {len(new_smiles):,}')

        # Subsample if dataset is too large
        if len(new_smiles) > args.max_per_dataset:
            print(f'Subsampling to {args.max_per_dataset:,} molecules')
            new_smiles = np.random.choice(new_smiles, size=args.max_per_dataset, replace=False).tolist()

        slices.append(slice(len(smiles), len(smiles) + len(new_smiles)))
        labels.append(label)
        smiles += new_smiles

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]

    print('Running t-SNE')
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    print(f'time = {time.time() - start:.2f} seconds')

    print('Plotting t-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    makedirs(args.save_path, isfile=True)

    plt.clf()
    fontsize = 50 * args.scale
    fig = plt.figure(figsize=(64 * args.scale, 48 * args.scale))
    #plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)
    ax = fig.gca()
    handles = []
    legend_kwargs = dict(loc='upper right', fontsize=fontsize)

    for slc, color, label, size in zip(slices, args.colors, labels, args.sizes):
        if args.plot_molecules:
            # Plots molecules
            handles.append(mpatches.Patch(color=color, label=label))
            for smile, (x, y) in zip(smiles[slc], X[slc]):
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color))
                    ax.add_artist(imagebox)
        else:
            # Plots points
            plt.scatter(X[slc, 0], X[slc, 1], s=150 * size, color=color, label=label)
            
    if args.plot_molecules:
        legend_kwargs['handles'] = handles

    plt.legend(**legend_kwargs)
    plt.xticks([]), plt.yticks([])

    print('Saving t-SNE')
    plt.savefig(args.save_path)


compare_datasets_tsne(Args)


