import os
import gc

from typing import Any

from enefit.preprocess.import_data import EnefitImport
from enefit.preprocess.add_feature import EnefitFeature
from enefit.preprocess.cv_fold import EnefitFoldCreator
from enefit.preprocess.initialization import EnefitInit

class EnefitPipeline(EnefitImport, EnefitFeature, EnefitFoldCreator):

    def __init__(self, config_dict: dict[str, Any], target_n_lags: int, agg_target_n_lags: int, embarko_skip: int):
                
        EnefitInit.__init__(
            self, 
            config_dict=config_dict, 
            target_n_lags=target_n_lags, agg_target_n_lags=agg_target_n_lags, 
            embarko_skip=embarko_skip
        )
        self.import_all()
        self.inference: bool = False
        
    def save_data(self) -> None:
        print('saving processed dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
    def create_feature(self) -> None:
        _ = gc.collect()
        self.copy_starting_dataset()
        
        self.create_client_feature()
        self.create_electricity_feature()
        self.create_forecast_weather_feature()
        self.create_gas_feature()
        self.create_historical_weather_feature()
        self.create_train_feature()
        self.create_target_feature()
        
    def preprocess_inference(self) -> None:
        self.create_feature()
        self.merge_all()
        
    def preprocess_train(self) -> None:
        self.create_feature()
        self.merge_all()
        
        print('Collecting....')
        self.data = self.data.collect()
        _ = gc.collect()
        
        print('Creating fold_info column ...')
        self.create_fold()
        self.save_data()
    
    def collect_all(self) -> None:
        self.location_data = self.location_data.collect()
        self.starting_client_data = self.starting_client_data.collect()
        self.starting_electricity_data = self.starting_electricity_data.collect()
        self.starting_forecast_weather_data = self.starting_forecast_weather_data.collect()
        self.starting_gas_data = self.starting_gas_data.collect()
        self.starting_historical_weather_data = self.starting_historical_weather_data.collect()
        self.starting_train_data = self.starting_train_data.collect()
        self.starting_target_data = self.starting_target_data.collect()
        
    def begin_inference(self) -> None:
        #reset data
        self.data = None
        self.inference: bool = True
        
        self.import_all()
        self.collect_all()
        self.create_feature()
        
    def __call__(self) -> None:
        if self.inference:
            self.preprocess_inference()
        else:
            self.preprocess_train()