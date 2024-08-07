import json
import os
import torch
import pickle
from typing import List, Optional, Tuple
from typing_extensions import Literal
from tempfile import TemporaryDirectory
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from data import set_cache_mol
from features import get_available_features_generators

Metric = Literal['acc', 'spear_corr']

class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    batch_size: int = 64
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""    
    features_generator: List[str] = None #['morgan_count'] #['rdkit_2d_normalized'] 
    """Choices=["morgan_count", "rdkit_2d_normalized"]
    Other custom features generator template is provided in features.py."""
    checkpoint_paths: List[str] = ['ROASMI_2.pt'] #['ROASMI_1.pt']
    """List of paths to model checkpoints (:code:`.pt` files)."""
    RT_interval: int = 25
    """The retention interval (in seconds) for generating training pairs."""
    gpu: int = None
    """Which GPU to use."""
    no_cache_mol: bool = False
    """Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default)."""
    quiet: bool = True
    """Skip non-essential print statements."""
    split_type: Literal['random', 'cv'] = 'random'
    """Method of splitting the data into train/validation/test."""
    num_folds: int = 10
    """Number of folds when performing cross validation."""
    warmup_epochs: float = 2.0
    """Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`."""
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss.""" 
    
    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')
        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    def add_arguments(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        set_cache_mol(not self.no_cache_mol)


class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    data_path: str = 'data/dataset_83.csv'
    """Path to reference data CSV files for (re)training."""
    separate_test_path: str = 'data/dataset_84.csv'
    """Path to external test set, optional."""
    config_path: str = None
    """Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults."""
    save_dir: str = '/output'
    """Directory where model checkpoints will be saved."""  
    MODEL_FILE_NAME: str = 'retrain_ROASMI.pt'
    """ Save file names"""
    TEST_SCORES_FILE_NAME: str = 'test_scores.csv'
    """ Save file names"""
    TRAIN_LOGGER_NAME: str = 'retrain'
    """Logger names"""
    SAVE_PATH_NAME: str = 'retrain.csv'
    epochs: int = 100
    """Number of epochs to run."""
    hidden_size: int = 1200
    """Dimensionality of hidden layers in MPNN."""
    depth: int = 6
    """Number of message passing steps."""
    dropout: float = 0.3
    """Dropout probability."""
    ff1_hidden_size: int = 800
    """Hidden dim for RankNet."""
    ff2_hidden_size: int = 300
    ffn_num_layers: int = 3
    """Number of layers in RankNet after MPNN encoding. Currently,only supported 1-3."""    
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors).
    Pay attention: Undirected is unnecessary when using atom_messages since atom_messages are by their nature undirected"""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network.
    Pay attention: When using features_only, a features_generator ust be provided."""
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    metric: Metric = 'acc'
    """Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "acc" for rank."""
    extra_metrics: List[Metric] = ['spear_corr']
    """Additional metrics to use to evaluate the model. """
    split_sizes: Tuple[float, float, float] = (0.9,0.1,0)
    """Split proportions for train/validation/test sets."""
    seed: int = 64
    """Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed."""
    pytorch_seed: int = 128
    """Seed for random initial weights."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """Maximum number of molecules in dataset to allow caching.Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.Use "inf" to always cache."""
    checkpoint_frzn: str = 'ROASMI.pt'  
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    frzn_ffn_layers: int = 0
    """Overwrites weights for the first n layers of the RankNet from checkpoint model (specified checkpoint_frzn), 
    where n is specified in the input. Automatically also freezes MPNN weights. """
    
    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._crossval_index_sets = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return self.features_generator is not None

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        # Prevents the temporary directory from being deleted upon function return
        global temp_dir  

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name
    
    
class HyperoptArgs(TrainArgs):
    """:class:`HyperoptArgs` includes :class:`TrainArgs` along with additional arguments used for optimizing Chemprop hyperparameters."""

    num_iters: int = 2
    """Number of hyperparameter choices to try."""
    config_save_path = '/output/hyperopt_choice.txt'
    """Path to .txt file where best hyperparameter settings will be written."""
    log_dir: str = '/output/hyperopt/'
    """(Optional) Path to a directory where all results of the hyperparameter optimization will be written.""" 
    HYPEROPT_LOGGER_NAME: str = 'hyperparameter-optimization'
    """Logger names"""

    
class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model.
       Pay attention: If features were used during training, they must be used when predicting."""

    TRAIN_LOGGER_NAME: str = 'predict'
    """Logger names"""
    SAVE_NAME: str = 'predict_result.csv'
    MODEL_FILE_NAME: str = 'retention_predict'
    """ Save file names"""
    pred_path: str = 'data/predict_toy_set.csv'
    """Path to CSV file containing ready-to-predict data."""
    preds_path: str = '/output/'
    """Path to CSV file where predictions will be saved."""
    save_dir: str = '/output/'
    """ Path to save logger."""   
    
    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

class InterpretArgs(CommonArgs):
    """:class:`InterpretArgs` includes :class:`CommonArgs` along with additional arguments used for interpreting a trained Chemprop model."""

    interpret_path: str = '/data/dataset_82.csv'
    """Path to data CSV file."""
    SAVE_PTAH_NAME = '/output/Interpret_82.csv'
    
    batch_size: int = 256
    """Batch size."""
    rollout: int = 20
    """Number of rollout steps."""
    c_puct: float = 10.0
    """Constant factor in MCTS."""
    max_atoms: int = 20
    """Maximum number of atoms in rationale."""
    min_atoms: int = 8
    """Minimum number of atoms in rationale."""
    prop_delta: float = 0.5
    """Minimum score to count as positive."""

    def process_args(self) -> None:
        super(InterpretArgs, self).process_args()

class IdentifyArgs(CommonArgs):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""
    margin_type: str = 'max'
    """Choices=["max", "sum"]; Which marginal should be used: max-marginal or sum-marginal."""
    n_trees: int = 256
    """Number of random spanning-trees to average the marginal distribution."""    
    D: float = 1
    """Scalar, weight on the retention order information."""
    identify_path: str = 'data/identify_toy_set.csv'
    """Path to data CSV file."""
    n_jobs: int = 8
    """Number of jobs used to parallelize the score-integration on the spanning-tree ensembles."""
    index_of_correct_structure: List[int] = None
    """An index of known analytes in the candidate list.Provided when known analytes are used to determine identification accuracy."""
