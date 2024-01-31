import numpy as np
import polars as pl

import xgboost as xgb
from src.model.xgbm.initialization import XgbInit

class XgbInference(XgbInit):     
    def load_feature_data(self, data: pl.DataFrame) -> np.ndarray:
        return data.select(self.feature_list).to_pandas().to_numpy(dtype='float32')
    
    def single_model_predict(self, test_data: pl.DataFrame) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
        prediction_ = self.array_single_model_predict(test_data=test_data)
        return prediction_
    
    def array_single_model_predict(self, test_data: np.ndarray) -> np.ndarray:
        prediction_ = self.model_all_data.predict(xgb.DMatrix(test_data))
        return prediction_
    
    def blend_model_predict(self, test_data: pl.DataFrame) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
        prediction_ = self.array_blend_model_predict(test_data=test_data)
        
        return prediction_
    
    def array_blend_model_predict(self, test_data: np.ndarray) -> np.ndarray:
        prediction_ = np.zeros((test_data.shape[0]), dtype='float64')
        
        for model in self.model_list:
            prediction_ += model.predict(
                xgb.DMatrix(test_data, feature_names=self.feature_list),
                iteration_range  = (0, self.best_result['best_epoch'])
            )/self.n_fold
            
        return prediction_
    
    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        assert self.inference
        
        if self.inference_setup == 'single':
            prediction_ = self.single_model_predict(test_data=test_data)
        
        elif self.inference_setup == 'blend':
            prediction_ = self.blend_model_predict(test_data=test_data)
        
        else:
            raise NotImplementedError
        
        return prediction_.clip(0)