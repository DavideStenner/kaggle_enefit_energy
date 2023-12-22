import os
import polars as pl

from typing import Any
from enefit.preprocess.initialization import EnefitInit

class EnefitImport(EnefitInit):    
    def scan_all_dataset(self) -> None:
        
        self.location_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.config_dict['PATH_MAPPING_DATA'], 'location_mapping.csv'
            )
        )
        
        self.client_data : pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'client.csv'
            )
        )
        self.train_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'train.csv'
            )
        )

        self.electricity_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 
                'electricity_prices.csv'
            )
        )
        self.gas_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'gas_prices.csv'
            )
        )
        self.forecast_weather_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'forecast_weather.csv'
            )
        )
        self.historical_weather_data: pl.LazyFrame = pl.scan_csv(
            os.path.join(
                self.path_original_data, 'historical_weather.csv'
            )
        )
        
    #CLIENT
    def downcast_client(self) -> None:
        self.client_data = self.client_data.with_columns(
            pl.col('product_type').cast(pl.UInt8),
            pl.col('county').cast(pl.UInt8),
            pl.col('eic_count').cast(pl.UInt16),
            pl.col('installed_capacity').cast(pl.Float32),
            pl.col('is_business').cast(pl.UInt8),
            pl.col('date').cast(pl.Date),
            pl.col('data_block_id').cast(pl.UInt16)
        )

    #GAS
    def downcast_gas_data(self) -> None:
        self.gas_data = self.gas_data.with_columns(
            pl.col('forecast_date').cast(pl.Date),
            pl.col('lowest_price_per_mwh').cast(pl.Float32),
            pl.col('highest_price_per_mwh').cast(pl.Float32),
            pl.col('origin_date').cast(pl.Date),
            pl.col('data_block_id').cast(pl.UInt16)
        )

    #ELECTRICITY
    def downcast_electricity(self) -> None:
        self.electricity_data = self.electricity_data.with_columns(
            (
                pl.col('forecast_date').str.to_datetime(),
                pl.col('euros_per_mwh').cast(pl.Float32),
                pl.col('origin_date').str.to_datetime(),
                pl.col('data_block_id').cast(pl.UInt16)
            )
        )
            
    #LOCATION
    def downcast_location(self) -> None:        
        self.location_data = self.location_data.with_columns(
            pl.col('county').cast(pl.UInt8),
            pl.col('longitude').cast(pl.Float32),
            pl.col('latitude').cast(pl.Float32)
        )
    
    #FORECAST WEATHER
    def downcast_forecast_weather_data(self) -> None:
        #downcast data
        self.forecast_weather_data = self.forecast_weather_data.with_columns(
            (
                pl.col('latitude').cast(pl.Float32),
                pl.col('longitude').cast(pl.Float32),
                pl.col('origin_datetime').str.to_datetime(),
                pl.col('hours_ahead').cast(pl.UInt8),
                pl.col('temperature').cast(pl.Float32),
                pl.col('dewpoint').cast(pl.Float32),
                pl.col('cloudcover_high').cast(pl.Float64),
                pl.col('cloudcover_low').cast(pl.Float64),
                pl.col('cloudcover_mid').cast(pl.Float64),
                pl.col('cloudcover_total').cast(pl.Float64),
                pl.col('10_metre_u_wind_component').cast(pl.Float64),
                pl.col('10_metre_v_wind_component').cast(pl.Float64),
                pl.col('data_block_id').cast(pl.UInt16),
                pl.col('direct_solar_radiation').cast(pl.Float64),
                pl.col('surface_solar_radiation_downwards').cast(pl.Float64),
                pl.col('snowfall').cast(pl.Float64),
                pl.col('total_precipitation').cast(pl.Float64),
                pl.col('forecast_datetime').str.to_datetime()
            )
        )
        
    #FORECAST WEATHER
    def downcast_historical_weather_data(self) -> None:        
        #downcast data
        self.historical_weather_data = self.historical_weather_data.with_columns(
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
                pl.col('data_block_id').cast(pl.UInt16),
            )
        )
        
    #TRAIN
    def downcast_train(self) -> None:
        self.train_data = self.train_data.with_columns(
            (
                pl.col('county').cast(pl.UInt8),
                pl.col('is_business').cast(pl.UInt8),
                pl.col('product_type').cast(pl.UInt8),
                pl.col('target').cast(pl.Float32),
                pl.col('is_consumption').cast(pl.UInt8),
                pl.col('datetime').str.to_datetime(),
                pl.col('data_block_id').cast(pl.UInt16),
                pl.col('row_id').cast(pl.UInt32),
                pl.col('prediction_unit_id').cast(pl.UInt8)
            )
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