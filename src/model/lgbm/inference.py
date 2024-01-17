import numpy as np
import polars as pl

from src.model.lgbm.initialization import LgbmInit

class LgbmInference(LgbmInit):     
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