import polars as pl
from typing import Any, Union, Dict

class EnefitInit():
    def __init__(self, 
            config_dict: dict[str, Any], target_n_lags: int, 
            embarko_skip: int
        ):
        
        self.target_n_lags: int = target_n_lags
        self.path_original_data: str = config_dict['PATH_ORIGINAL_DATA']
        self.config_dict: dict[str, Any] = config_dict
        self.embarko_skip: int = embarko_skip
        self.n_folds: int = config_dict['N_FOLD']
        self.fold_time_col: str = 'date_order_kfold'
        self.inference: bool = False

        self._initialiaze_empty_dataset()
        self._initialize_used_column()
        self._initialize_pandas_schema()
        
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
        
        self.main_data: Union[pl.LazyFrame, pl.DataFrame] = None        

        self.target_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.starting_target_data: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.data: Union[pl.LazyFrame, pl.DataFrame] = None
    
    def _initialize_pandas_schema(self) -> None:
        self.pandas_dataset_schema_dict: Dict[str, Dict[str, str]] = {
            'client': {
                'product_type': 'int8',
                'county': 'int8',
                'eic_count': 'int16',
                'installed_capacity': 'float32',
                'is_business': 'bool'
            },
            'electricity': {
                'euros_per_mwh': 'float32',
            },
            'forecast_weather': {
                'latitude': 'float32',
                'longitude': 'float32',
                'hours_ahead': 'int8',
                'temperature': 'float32',
                'dewpoint': 'float32',
                'cloudcover_high': 'float64',
                'cloudcover_low': 'float64',
                'cloudcover_mid': 'float64',
                'cloudcover_total': 'float64',
                '10_metre_u_wind_component': 'float64',
                '10_metre_v_wind_component': 'float64',
                'direct_solar_radiation': 'float64',
                'surface_solar_radiation_downwards': 'float64',
                'snowfall': 'float64',
                'total_precipitation': 'float64'
            },
            'gas': {
                'lowest_price_per_mwh': 'float32',
                'highest_price_per_mwh': 'float32',
            },
            'historical_weather': {
                'temperature': 'float32',
                'dewpoint': 'float32',
                'rain': 'float32',
                'snowfall': 'float32',
                'surface_pressure': 'float32',
                'cloudcover_total': 'int32',
                'cloudcover_low': 'int32',
                'cloudcover_mid': 'int32',
                'cloudcover_high': 'int32',
                'windspeed_10m': 'float32',
                'winddirection_10m': 'int32',
                'shortwave_radiation': 'int32',
                'direct_solar_radiation': 'int32',
                'diffuse_radiation': 'int32',
                'latitude': 'float32',
                'longitude': 'float32'
            },
            'test': {
                'county': 'int8',
                'is_business': 'bool',
                'product_type': 'int8',
                'is_consumption': 'bool',
                'row_id': 'int32',
                'prediction_unit_id': 'int16',
                'currently_scored': 'bool'
            },
            'target': {
                'county': 'int8',
                'is_business': 'bool',
                'product_type': 'int8',
                'target': 'float64',
                'is_consumption': 'bool',
                'row_id': 'int32',
                'prediction_unit_id': 'int16'
            }
        }

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
            ],
            'target': [
                "target",
                "county",
                "is_business",
                "product_type",
                "is_consumption",
                "datetime",
            ],
            'test': [
                "county",
                "is_business",
                "product_type",
                "is_consumption",
                "datetime",
                "row_id",
            ] 
        }

    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        return data.item() if self.inference else data.collect().item()