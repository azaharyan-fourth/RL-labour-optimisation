import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, file_path, location_id=None):
        df = pd.read_csv(file_path)
        
        if location_id != None:
            df = df[df['department_id'] == location_id]

        self.series = df

        self.split_train = round(0.8*df.size)
        self.split_dataset()

    def split_dataset(self):
        #train-test split dataset
        self.dataset_train = self.series[:self.split_train]
        self.dataset_val = self.series[self.split_train+1:]

    

      
