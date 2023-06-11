from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    # args.smap = dataset['smap']
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
