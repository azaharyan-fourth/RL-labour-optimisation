import numpy as np
import pandas as pd
from datetime import datetime

class Dataset:
        
    def __init__(self, 
                file_path: str, 
                start_test_period: str):
        df = pd.read_csv(file_path,
                            parse_dates=['date'],
                            date_parser=lambda x: datetime.strptime(str(x), '%Y-%m-%d'))


        df.drop('index', axis=1, inplace=True)
        df = df.reset_index(drop=True)
        df.index.name = 'index'
        df = df.sort_index()
        self.series = df
        
        self.split_dataset(start_test_period)

    def split_dataset(self, start_test_period):
        #train-test split dataset
        self.dataset_train = self.series[self.series['date'] < start_test_period]
        #self.dataset_val = self.series[(self.series['date'] >= start_test_period)]
        val_test = np.array_split(self.series[(self.series['date'] >= start_test_period)], 2)
        self.dataset_val = val_test[0]
        self.dataset_test = val_test[1]