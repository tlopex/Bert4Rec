from .base import AbstractDataset2

import pandas as pd

from datetime import date


class KDD_Dataset(AbstractDataset2):
    @classmethod
    def code(cls):
        return 'KDD'

    @classmethod
    def url(cls):
        pass
        # return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        # return True
        pass

    @classmethod
    def all_raw_file_names(cls):
        # return ['history.dat','history_mini.dat']
        pass

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.args.data_path)
        # file_path = folder_path.joinpath('JP_data_mini.dat')
        # file_path = folder_path.joinpath('DE_data_mini.dat')
        
        df = pd.read_csv(file_path, sep='::', header=None)
        
        df.columns = ['uid', 'sid', 'price', 'brand', 'color', 'author', 'category', 'timestamp']

        return df


