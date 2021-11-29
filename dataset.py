import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, file_path, location_id=None):
        df = pd.read_csv(f"{file_path}/all_normal_shifts_{location_id}.csv")

        df = df.set_index('index')
        df = df.sort_index()
        self.series = df

        #self.split_train = round(0.8*df.size)
        self.split_dataset()

    def split_dataset(self):
        #train-test split dataset
        self.dataset_train = self.series[self.series['date'] < '2020-01-01']
        self.dataset_val = self.series[(self.series['date'] >= '2020-01-01') &
                             (self.series['date'] < '2020-03-01')]

    

      
