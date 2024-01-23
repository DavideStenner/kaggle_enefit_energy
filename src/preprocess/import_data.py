import os
import polars as pl
import pandas as pd

from typing import Dict, OrderedDict, List
from src.preprocess.initialization import EnefitInit

class EnefitImport(EnefitInit):    
    def set_type_new_data(
            self,
            client_data_new: pd.DataFrame,
            gas_data_new: pd.DataFrame,
            electricity_data_new: pd.DataFrame,
            forecast_weather_data_new: pd.DataFrame,
            historical_weather_data_new: pd.DataFrame,
            target_data_new: pd.DataFrame,
            test_data: pd.DataFrame
    ) -> List[pd.DataFrame]:
        
        #TEST
        test_data = test_data.astype(self.pandas_dataset_schema_dict['test'])
        test_data['prediction_datetime'] = pd.to_datetime(test_data['prediction_datetime'])
        test_data = test_data.rename(
            columns={"prediction_datetime": "datetime"}
        )
        
        #CLIENT
        client_data_new = client_data_new.astype(self.pandas_dataset_schema_dict['client'])
        client_data_new['date'] = pd.to_datetime(client_data_new['date'])
        
        #GAS
        gas_data_new = gas_data_new.astype(self.pandas_dataset_schema_dict['gas'])
        
        gas_data_new['origin_date'] = pd.to_datetime(gas_data_new['origin_date'])
        gas_data_new['forecast_date'] = pd.to_datetime(gas_data_new['forecast_date'])

        #ELECTRICITY
        electricity_data_new = electricity_data_new.astype(self.pandas_dataset_schema_dict['electricity'])
        electricity_data_new['origin_date'] = pd.to_datetime(electricity_data_new['origin_date'])
        electricity_data_new['forecast_date'] = pd.to_datetime(electricity_data_new['forecast_date'])

        #FORECAST WEATHER
        forecast_weather_data_new = forecast_weather_data_new.astype(
            self.pandas_dataset_schema_dict['forecast_weather']
        )
        forecast_weather_data_new['origin_datetime'] = pd.to_datetime(forecast_weather_data_new['origin_datetime'])
        forecast_weather_data_new['forecast_datetime'] = pd.to_datetime(forecast_weather_data_new['forecast_datetime'])
       
        #HISTORICAL WEATHER
        historical_weather_data_new = historical_weather_data_new.astype(
            self.pandas_dataset_schema_dict['historical_weather']
        )
        historical_weather_data_new['datetime'] = pd.to_datetime(historical_weather_data_new['datetime'])

        #TARGE
        target_data_new = target_data_new.astype(
            self.pandas_dataset_schema_dict['target']
        )
        target_data_new['datetime'] = pd.to_datetime(target_data_new['datetime'])
        return (
            client_data_new,
            gas_data_new,
            electricity_data_new,
            forecast_weather_data_new,
            historical_weather_data_new,
            target_data_new,
            test_data
        )
        
    def update_with_new_data(
            self,
            client_data_new: pd.DataFrame,
            gas_data_new: pd.DataFrame,
            electricity_data_new: pd.DataFrame,
            forecast_weather_data_new: pd.DataFrame,
            historical_weather_data_new: pd.DataFrame,
            target_data_new: pd.DataFrame,
            test_data: pd.DataFrame
        ) -> None:
        
        if not self.inference:
            raise ValueError('Call begin_inference first...')
        
        #ensure new data has correct dtype
        (
            client_data_new,
            gas_data_new,
            electricity_data_new,
            forecast_weather_data_new,
            historical_weather_data_new,
            target_data_new,
            test_data
        ) = self.set_type_new_data(
            client_data_new,
            gas_data_new,
            electricity_data_new,
            forecast_weather_data_new,
            historical_weather_data_new,
            target_data_new,
            test_data
        )
        
        test_data = pl.from_pandas(
            test_data[self.starting_dataset_column_dict['test']], 
            schema_overrides=self.starting_dataset_schema_dict['train']
        )
        client_data_new = pl.from_pandas(
            client_data_new[self.starting_dataset_column_dict['client']], 
            schema_overrides=self.starting_dataset_schema_dict['client']
        )
        gas_data_new = pl.from_pandas(
            gas_data_new[self.starting_dataset_column_dict['gas']],
            schema_overrides=self.starting_dataset_schema_dict['gas'],
        )
        electricity_data_new = pl.from_pandas(
            electricity_data_new[self.starting_dataset_column_dict['electricity']],
            schema_overrides=self.starting_dataset_schema_dict['electricity'],
        )
        forecast_weather_data_new = pl.from_pandas(
            forecast_weather_data_new[self.starting_dataset_column_dict['forecast_weather']],
            schema_overrides=self.starting_dataset_schema_dict['forecast_weather'],
        )
        historical_weather_data_new = pl.from_pandas(
            historical_weather_data_new[self.starting_dataset_column_dict['historical_weather']],
            schema_overrides=self.starting_dataset_schema_dict['historical_weather'],
        )
        target_data_new = pl.from_pandas(
            target_data_new[self.starting_dataset_column_dict['target']], 
            schema_overrides=self.starting_dataset_schema_dict['target']
        )

        self.main_data = test_data
        self.starting_client_data = pl.concat(
            [self.starting_client_data, client_data_new]
        ).unique(
            ["date", "county", "is_business", "product_type"]
        )
        self.starting_gas_data = pl.concat(
            [self.starting_gas_data, gas_data_new]
        ).unique(
            ["forecast_date"]
        )
        self.starting_electricity_data = pl.concat(
            [self.starting_electricity_data, electricity_data_new]
        ).unique(["forecast_date"])
        
        self.starting_forecast_weather_data = pl.concat(
            [self.starting_forecast_weather_data, forecast_weather_data_new]
        ).unique(["origin_datetime", "hours_ahead", "latitude", "longitude", "hours_ahead"])
        
        self.starting_historical_weather_data = pl.concat(
            [self.starting_historical_weather_data, historical_weather_data_new]
        ).unique(["datetime", "latitude", "longitude"])
        
        self.starting_target_data = pl.concat([self.starting_target_data, target_data_new]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]
        )

     
    def scan_all_dataset(self) -> None:
        
        self.location_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.config_dict['PATH_MAPPING_DATA'], 'location_mapping.csv'
            )
        )
        
        self.starting_client_data : pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'client.csv'
            )
        )
        self.main_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'train.csv'
            )
        )

        self.starting_electricity_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 
                'electricity_prices.csv'
            )
        )
        self.starting_gas_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'gas_prices.csv'
            )
        )
        self.starting_forecast_weather_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'forecast_weather.csv'
            )
        )
        self.starting_historical_weather_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'historical_weather.csv'
            )
        )
        
    #CLIENT
    def downcast_client(self) -> None:
        self.starting_client_data = self.starting_client_data.select(
            self.starting_dataset_column_dict['client']
        ).with_columns(
            pl.col('product_type').cast(pl.UInt8),
            pl.col('county').cast(pl.UInt8),
            pl.col('eic_count').cast(pl.UInt16),
            pl.col('installed_capacity').cast(pl.Float32),
            pl.col('is_business').cast(pl.UInt8),
            pl.col('date').cast(pl.Date)
        )

    #GAS
    def downcast_gas_data(self) -> None:
        self.starting_gas_data = self.starting_gas_data.select(
            self.starting_dataset_column_dict['gas']
        ).with_columns(
            pl.col('forecast_date').str.to_datetime(),
            pl.col('lowest_price_per_mwh').cast(pl.Float32),
            pl.col('highest_price_per_mwh').cast(pl.Float32),
        )

    #ELECTRICITY
    def downcast_electricity(self) -> None:
        self.starting_electricity_data = self.starting_electricity_data.select(
            self.starting_dataset_column_dict['electricity']
        ).with_columns(
            (
                pl.col('forecast_date').str.to_datetime(),
                pl.col('euros_per_mwh').cast(pl.Float32)
            )
        )
            
    #LOCATION
    def downcast_location(self) -> None:        
        self.location_data = self.location_data.select(
            self.starting_dataset_column_dict['location']
        ).with_columns(
            pl.col('county').cast(pl.UInt8),
            pl.col('longitude').cast(pl.Float32),
            pl.col('latitude').cast(pl.Float32)
        )
    
    #FORECAST WEATHER
    def downcast_forecast_weather_data(self) -> None:
        #downcast data
        self.starting_forecast_weather_data = self.starting_forecast_weather_data.select(
            self.starting_dataset_column_dict['forecast_weather']
        ).with_columns(
            (
                pl.col('latitude').cast(pl.Float32),
                pl.col('longitude').cast(pl.Float32),
                pl.col('hours_ahead').cast(pl.UInt8),
                pl.col('temperature').cast(pl.Float32),
                pl.col('dewpoint').cast(pl.Float32),
                pl.col('cloudcover_high').cast(pl.Float64),
                pl.col('cloudcover_low').cast(pl.Float64),
                pl.col('cloudcover_mid').cast(pl.Float64),
                pl.col('cloudcover_total').cast(pl.Float64),
                pl.col('10_metre_u_wind_component').cast(pl.Float64),
                pl.col('10_metre_v_wind_component').cast(pl.Float64),
                pl.col('direct_solar_radiation').cast(pl.Float64),
                pl.col('surface_solar_radiation_downwards').cast(pl.Float64),
                pl.col('snowfall').cast(pl.Float64),
                pl.col('total_precipitation').cast(pl.Float64),
                pl.col('origin_datetime').str.to_datetime()
            )
        )
        
    #FORECAST WEATHER
    def downcast_historical_weather_data(self) -> None:        
        #downcast data
        self.starting_historical_weather_data = self.starting_historical_weather_data.select(
            self.starting_dataset_column_dict['historical_weather']
        ).with_columns(
            (
                pl.col('datetime').str.to_datetime(),
                pl.col('temperature').cast(pl.Float32),
                pl.col('dewpoint').cast(pl.Float32),
                pl.col('rain').cast(pl.Float32),
                pl.col('snowfall').cast(pl.Float32),
                pl.col('surface_pressure').cast(pl.Float32),
                pl.col('cloudcover_total').cast(pl.UInt8),
                pl.col('cloudcover_low').cast(pl.UInt8),
                pl.col('cloudcover_mid').cast(pl.UInt8),
                pl.col('cloudcover_high').cast(pl.UInt8),
                pl.col('windspeed_10m').cast(pl.Float32),
                pl.col('winddirection_10m').cast(pl.UInt16),
                pl.col('shortwave_radiation').cast(pl.UInt16),
                pl.col('direct_solar_radiation').cast(pl.UInt16),
                pl.col('diffuse_radiation').cast(pl.UInt16),
                pl.col('latitude').cast(pl.Float32),
                pl.col('longitude').cast(pl.Float32),
            )
        #take out duplicates from train
        ).unique(
            ['latitude', 'longitude', 'datetime']
        )
        
    #TRAIN
    def downcast_train(self) -> None:
        self.main_data = self.main_data.select(
            self.starting_dataset_column_dict['train']
        ).with_columns(
            (
                pl.col('county').cast(pl.UInt8),
                pl.col('is_business').cast(pl.UInt8),
                pl.col('product_type').cast(pl.UInt8),
                pl.col('target').cast(pl.Float64),
                pl.col('is_consumption').cast(pl.UInt8),
                pl.col('datetime').str.to_datetime(),
                pl.col('row_id').cast(pl.UInt32)
            )
        )

    def filter_train(self) -> None:
        self.main_data = self.main_data.filter(
            pl.col("datetime") >= pd.to_datetime("2022-01-01")
        )
        
    def memorize_starting_dataset_schema(self) -> None:
        self.starting_dataset_schema_dict: Dict[str, OrderedDict] = {
            'client': self.starting_client_data.schema,
            'electricity': self.starting_electricity_data.schema,
            'forecast_weather': self.starting_forecast_weather_data.schema,
            'gas': self.starting_gas_data.schema,
            'historical_weather': self.starting_historical_weather_data.schema,
            'location': self.location_data.schema,
            'train': self.main_data.schema,
            'target': self.starting_target_data.schema
        }
    
    def create_target_data(self) -> None:
        self.starting_target_data: pl.LazyFrame = self.main_data.select(
            self.starting_dataset_column_dict['target']
        )

    
    def import_all(self) -> None:
        self.scan_all_dataset()
        
        self.downcast_client()
        self.downcast_electricity()
        self.downcast_forecast_weather_data()
        self.downcast_gas_data()
        self.downcast_historical_weather_data()
        self.downcast_location()
        self.downcast_train()
        
        self.filter_train()
        
        self.create_target_data()
        
        self.memorize_starting_dataset_schema()