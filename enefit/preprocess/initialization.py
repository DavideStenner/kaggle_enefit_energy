import polars as pl
from typing import Any, Union

class EnefitInit():
    def __init__(self, 
            config_dict: dict[str, Any], target_n_lags: int, 
            agg_target_n_lags: list[int], embarko_skip: int
        ):
        
        self.target_n_lags: int = target_n_lags
        self.agg_target_n_lags: int = agg_target_n_lags
        self.path_original_data: str = config_dict['PATH_ORIGINAL_DATA']
        self.config_dict: dict[str, Any] = config_dict
        self.embarko_skip: int = embarko_skip
        self.n_folds: int = config_dict['N_FOLD']

        self._initialiaze_empty_dataset()
        
    def _initialiaze_empty_dataset(self):
        self.client_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_client_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.electricity_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_electricity_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.forecast_weather_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_forecast_weather_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.gas_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_gas_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.historical_weather_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_historical_weather_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.location_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.train_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_train_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.data: Union[pl.LazyFrame, pl.DataFrame] = None