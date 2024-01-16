import os
import json
import pickle

import numpy as np
import polars as pl

from enefit.model.lgbm.initialization import LgbmInit

class LgbmInference(LgbmInit):     
    def load_model(self) -> None:  
        #load feature list
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'r'
        ) as file:
            self.feature_list = json.load(file)
            
        with open(
            os.path.join(
                self.experiment_path,
                'best_result_lgb.txt'
            ), 'r'
        ) as file:
            self.best_result = json.load(file)
            
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'rb'
        ) as file:
            self.model_list = pickle.load(file)

        self.inference = True
    def predict(self, test_data: pl.DataFrame) -> None:
        assert self.inference
        
        test_data = test_data.select(self.feature_list).to_numpy().astype('float32')
        
        prediction_ = np.zeros((test_data.shape[0]))
        
        for model in self.model_list:
            prediction_ += model.predict(
                test_data,
                num_iteration = self.best_result['best_epoch']
            )/self.n_fold
            
        return prediction_