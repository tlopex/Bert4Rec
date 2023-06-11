import torch
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import pandas as pd


        

def train():
    
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    print(args.num_items)
    model = model_factory(args)
    model.load_state_dict(torch.load(args.para_path_uk))
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    # trainer.train()
    # torch.save(model.state_dict(), args.para_path)
    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.output_list()
    
def get_ans():
    res = []
    
    preprocess_path = "D:\BERT4Rec\BERT4Rec-VAE-Pytorch-master\Data\preprocessed"
    
    
    args.is_val= 0
    args.device = 'cuda'
    if os.path.exists(preprocess_path):
        shutil.rmtree(preprocess_path)
    export_root = setup_train(args)
    args.data_path = 'DE_data_mini.dat'
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    model.load_state_dict(torch.load(args.para_path_de))
    args.is_val = 1
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    temp = trainer.output_list()
    print(len(temp))
    res.extend(temp)
    
    # args.is_val= 0
    # if os.path.exists(preprocess_path):
    #     shutil.rmtree(preprocess_path)
    # export_root = setup_train(args)
    # args.data_path = 'JP_data_mini.dat'
    # train_loader, val_loader, test_loader = dataloader_factory(args)
    # model = model_factory(args)
    # model.load_state_dict(torch.load(args.para_path_jp))
    # args.is_val = 1
    # trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    # temp = trainer.output_list()
    # print(len(temp))
    # res.extend(temp)
    
    # args.is_val = 0
    # if os.path.exists(preprocess_path):
    #     shutil.rmtree(preprocess_path)
    # export_root = setup_train(args)
    # args.data_path = 'UK_data_mini.dat'
    # train_loader, val_loader, test_loader = dataloader_factory(args)
    # model = model_factory(args)
    # model.load_state_dict(torch.load(args.para_path_uk))
    # args.is_val = 1
    # trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    # temp = trainer.output_list()
    # print(len(temp))
    # res.extend(temp)
    
    # print(res)
    data = pd.read_csv("result.csv")
    next_item_series = pd.Series(res, name="next_item_prediction")
    data_with_next_item = pd.concat([data, next_item_series], axis=1)
    data_with_next_item.to_csv("data_with_next_item.csv", index=False)
    
    
    df = pd.DataFrame({'next_item_prediction': res})
    # df.to_csv('result.csv', mode='a', header=False, index=False)


    table = pa.Table.from_pandas(df)
    pq.write_table(table, 'result.parquet')
    

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'get_ans':
        get_ans()
    else:
        raise ValueError('Invalid mode')
