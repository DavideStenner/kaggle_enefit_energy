import polars as pl
from typing import Any, Union, Dict, OrderedDict

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
        self.fold_time_col: str = 'date_order_kfold'
        
        self._initialiaze_empty_dataset()
        self._initialize_used_column()
        
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
    
    def _initialize_used_column(self) -> None:
        self.starting_dataset_column_dict: Dict[str, list[str]] = {
            'client': [
                "product_type",
                "county",
                "eic_count",
                "installed_capacity",
                "is_business",
                "date"
            ],
            'electricity': ["forecast_date", "euros_per_mwh"],
            'forecast_weather': [
                "latitude",
                "longitude",
                "hours_ahead",
                "temperature",
                "dewpoint",
                "cloudcover_high",
                "cloudcover_low",
                "cloudcover_mid",
                "cloudcover_total",
                "10_metre_u_wind_component",
                "10_metre_v_wind_component",
                "origin_datetime",
                "direct_solar_radiation",
                "surface_solar_radiation_downwards",
                "snowfall",
                "total_precipitation",
            ],
            'gas': [
                "forecast_date", 
                "lowest_price_per_mwh", 
                "highest_price_per_mwh"
            ],
            'historical_weather': [
                "datetime",
                "temperature",
                "dewpoint",
                "rain",
                "snowfall",
                "surface_pressure",
                "cloudcover_total",
                "cloudcover_low",
                "cloudcover_mid",
                "cloudcover_high",
                "windspeed_10m",
                "winddirection_10m",
                "shortwave_radiation",
                "direct_solar_radiation",
                "diffuse_radiation",
                "latitude",
                "longitude",
            ],
            'location': ["longitude", "latitude", "county"],
            'train': [
                "county",
                "is_business",
                "product_type",
                "target",
                "is_consumption",
                "datetime",
                "row_id",
                "prediction_unit_id"
            ]
        }