from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .KDD_dataset import KDD_Dataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    KDD_Dataset.code() : KDD_Dataset
}


def dataset_factory(args):
    # dataset = DATASETS[args.dataset_code]
    dataset = KDD_Dataset
    
    return dataset(args)
