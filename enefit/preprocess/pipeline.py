import os
import gc

from typing import Any

from enefit.preprocess.import_data import EnefitImport
from enefit.preprocess.add_feature import EnefitFeature
from enefit.preprocess.cv_fold import EnefitFoldCreator
from enefit.preprocess.initialization import EnefitInit

class EnefitPipeline(EnefitImport, EnefitFeature, EnefitFoldCreator):

    def __init__(self, config_dict: dict[str, Any], target_n_lags: int):
                
        EnefitInit.__init__(self, config_dict=config_dict, target_n_lags=target_n_lags)
        self.import_all()
    
    def save_data(self) -> None:
        print('saving processed dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
        
    def __call__(self) -> None:
        _ = gc.collect()
        
        print('Creating Feature')
        self.create_client_feature()
        self.create_electricity_feature()
        self.create_forecast_weather_feature()
        self.create_gas_feature()
        self.create_historical_weather_feature()
        self.create_train_feature()
        
        print('Merging everything')
        self.merge_all()
        
        print('Collecting....')
        self.data = self.data.collect()
        _ = gc.collect()
        
        print('Creating fold_info column ...')
        self.create_fold()
        self.save_data()