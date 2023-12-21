import gc

from typing import Any

from enefit.preprocess.import_data import EnefitImport
from enefit.preprocess.add_feature import EnefitFeature

class EnefitPipeline(EnefitImport, EnefitFeature):

    def __init__(self, config_dict: dict[str, Any], target_n_lags: int):
        
        self.target_n_lags = target_n_lags
        
        EnefitImport.__init__(self, config_dict=config_dict)
        self.import_all()
    
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
