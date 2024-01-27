import numpy as np
import polars as pl
import holidays

from itertools import product
from datetime import timedelta
from typing import Dict, Union
from src.preprocess.initialization import EnefitInit

class EnefitFeature(EnefitInit):
    
    def copy_starting_dataset(self) -> None:
        #keep starting dataset without modification -> used during inference to update it
        self.client_data = self.starting_client_data
        self.gas_data = self.starting_gas_data
        self.electricity_data = self.starting_electricity_data
        self.forecast_weather_data = self.starting_forecast_weather_data
        self.historical_weather_data = self.starting_historical_weather_data
        self.target_data = self.starting_target_data
        
    def create_client_feature(self) -> None:
        #add new column
        self.client_data = self.client_data.with_columns(
            #clean date column
            (pl.col("date") + pl.duration(days=2)).cast(pl.Date),
            #eic count
            (
                pl.col('eic_count').mean()
                .over(['date', 'product_type', 'county', 'is_business'])
                .cast(pl.UInt16)
                .alias('eic_count_mean_date')
            ),
            (
                pl.col('eic_count').mean()
                .over(['product_type', 'county', 'is_business'])
                .cast(pl.UInt16)
                .alias('eic_count_mean')
            ),
            #installed capacity
            (
                pl.col('installed_capacity').mean()
                .over(['date', 'product_type', 'county', 'is_business'])
                .cast(pl.Float32)
                .alias('installed_capacity_mean_date')
            ),
            (
                pl.col('installed_capacity').mean()
                .over(['product_type', 'county', 'is_business'])
                .cast(pl.Float32)
                .alias('installed_capacity_mean')
            ),
            #log 1p installed capacity
            (
                pl.col('installed_capacity').log1p()
                .cast(pl.Float32)
                .alias('installed_capacity_log1p')
            ),
            (
                pl.col('installed_capacity').log1p().mean()
                .over(['date', 'product_type', 'county', 'is_business'])
                .cast(pl.Float32)
                .alias('installed_capacity_log1p_mean_date')
            ),
            (
                pl.col('installed_capacity').log1p().mean()
                .over(['product_type', 'county', 'is_business'])
                .cast(pl.Float32)
                .alias('installed_capacity_log1p_mean')
            )
        )
        
    def create_gas_feature(self) -> None:
        self.gas_data = self.gas_data.with_columns(
            (
                (pl.col('forecast_date') + pl.duration(days=1))
                .dt.date().alias('date')
                .cast(pl.Date)
            ),
            (
                (
                    pl.col('lowest_price_per_mwh') + 
                    pl.col('highest_price_per_mwh')
                )/2
            ).alias('mean_price_per_mwh').cast(pl.Float32),
            (
                pl.col('highest_price_per_mwh')-
                pl.col('lowest_price_per_mwh')
            ).alias('range_price_per_mwh').cast(pl.Float32)
        )
        self.gas_data = self.gas_data.drop(['forecast_date'])
        
    def create_electricity_feature(self) -> None:
        self.electricity_data = self.electricity_data.with_columns(
            (pl.col("forecast_date") + pl.duration(days=1)).alias('datetime')
        )
        #not needed during merge
        self.electricity_data = self.electricity_data.drop('forecast_date')
            
    def create_forecast_weather_feature(self) -> None:
        min_hour, max_hour = 22, 45
        
        #filter hours ahead
        self.forecast_weather_data = self.forecast_weather_data.filter(
            (pl.col("hours_ahead") >= min_hour) & 
            (pl.col("hours_ahead") <= max_hour)
        )
        #add date which is a key
        self.forecast_weather_data = self.forecast_weather_data.with_columns(
            (pl.col('origin_datetime') + pl.duration(days=1)).cast(pl.Date).alias('date')
        )

        training_variable: list[str] = [
            'temperature', 'dewpoint', 
            'cloudcover_high', 'cloudcover_low', 
            'cloudcover_mid', 'cloudcover_total', 
            '10_metre_u_wind_component', 
            '10_metre_v_wind_component', 
            'direct_solar_radiation', 
            'surface_solar_radiation_downwards', 
            'snowfall', 'total_precipitation'
        ]
        index_variable: list[str] = [
            'county', 'date'
        ]
    
        #add county
        self.forecast_weather_data = self.forecast_weather_data.join(
            self.location_data, on=['latitude', 'longitude']
        ).filter(pl.col('county').is_not_null())
        
        self.forecast_weather_data = self.forecast_weather_data.filter(pl.col("county").is_not_null())
        
        #PIVOT
        original_col_dict = {
            col: self.forecast_weather_data.select(col).dtypes[0]
            for col in training_variable
        }
        combination_pivot = list(product(training_variable, list(range(min_hour, max_hour+1))))

        self.forecast_weather_data = (
            self.forecast_weather_data
            .group_by(pl.col(index_variable))
            .agg(
                *[
                    (
                        (
                            pl.col(train_col)
                            .filter(pl.col('hours_ahead') == hours)
                        ).mean()
                        .alias(f'{train_col}_hours_ahead_{hours}')
                        .cast(original_col_dict[train_col]) 
                    )
                    for train_col, hours in combination_pivot
                ]
            )
        )
        
    def create_historical_weather_feature(self) -> None:
        min_hour, max_hour = 0, 23
        #add date which is a key
        self.historical_weather_data = self.historical_weather_data.with_columns(
            (
                pl.when(
                    pl.col('datetime').dt.hour()<11
                ).then(
                    pl.col('datetime') + pl.duration(days=1)
                ).otherwise(
                    pl.col('datetime') + pl.duration(days=2)
                )
            ).cast(pl.Date).alias('date')
        )
        training_variable: list[str] = [
            'temperature', 'dewpoint', 'rain',
            'snowfall', 'surface_pressure', 'cloudcover_total',
            'cloudcover_low', 'cloudcover_mid',
            'cloudcover_high', 'windspeed_10m',
            'winddirection_10m', 'shortwave_radiation',
            'direct_solar_radiation', 'diffuse_radiation'
        ]
        index_variable: list[str] = [
            'county', 'date'
        ]
        
        #add county
        self.historical_weather_data = self.historical_weather_data.join(
            self.location_data, on=['latitude', 'longitude']
        ).filter(pl.col('county').is_not_null())
    
        self.historical_weather_data = self.historical_weather_data.filter(pl.col("county").is_not_null())
        
        self.historical_weather_data = (
            self.historical_weather_data
            .sort(
                ['latitude', 'longitude', 'date', 'datetime'],
                descending=[False, False, False, True]
            )
            .with_columns(
                pl.arange(0, pl.count())
                .over(['latitude', 'longitude', 'date'])
                .cast(pl.UInt8).alias('hours_ago')
            )
        )
        
        #PIVOT
        original_col_dict = {
            col: self.historical_weather_data.select(col).dtypes[0]
            for col in training_variable
        }
        combination_pivot = list(product(training_variable, list(range(min_hour, max_hour+1))))
        self.historical_weather_data = (
            self.historical_weather_data
            .group_by(pl.col(index_variable))
            .agg(
                *[
                    (
                        (
                        pl.col(train_col)
                        .filter(pl.col('hours_ago') == hours)
                        ).mean()
                        .alias(f'{train_col}_hours_ago_{hours}')
                        .cast(original_col_dict[train_col])
                    )
                    for train_col, hours in combination_pivot
                ]
            )
        )

    def create_target_feature(self) -> None:
        key_list: list[str] = ['county', 'is_business', 'product_type', 'is_consumption']

        #this dataset is used to calculate the lag
        target_data = self.target_data.with_columns(
            pl.col('datetime').dt.date().alias('date').cast(pl.Date)
        )
        
        min_datetime = self._collect_item_utils(
            target_data.select('datetime').min()
        )
        max_datetime = self._collect_item_utils(
            target_data.select('datetime').max()
        )
        
        #create join target df -> create more date so i can get all usable lag
        #fix to main bugc
        datetime_explosion = pl.datetime_range(
            min_datetime - timedelta(days=self.target_n_lags+10),
            max_datetime + timedelta(days=self.target_n_lags+10),
            timedelta(hours=1), eager=True
        ).to_frame('datetime')
        
        if not self.inference:
            datetime_explosion = datetime_explosion.lazy()
            
        #this dataset is used to join to main_data
        target_feature = (
            target_data.select(key_list).unique()
            .join(
                datetime_explosion, how='cross'
            ).with_columns(
                pl.col('datetime').dt.date().alias('date').cast(pl.Date)
            )
        )
        # #add lag with aggregation on date
        aggregation_by_date = (
            target_data.select(
                key_list + ['date', 'target']
            ).group_by(key_list + ['date'])
            .agg(
                pl.col('target').mean().cast(pl.Float32)
            )
        )
        
        
        aggregation_by_dict: Dict[str, Union[pl.LazyFrame, pl.DataFrame]] = {}
        join_by_dict: Dict[str, list] = {}   
        
        for col in key_list:
            other_key_list = [x for x in key_list if x != col]
            aggregation_by_dict[col] = (
                target_data.select(
                    other_key_list + ['datetime', 'target']
                ).group_by(other_key_list + ['datetime'])
                .agg(
                    pl.col('target').mean().cast(pl.Float32)
                )
            )
            join_by_dict[col] = other_key_list
            
        #add all lag information
        for day_lag in range(2, self.target_n_lags+1):
            #add normal value
            target_feature = target_feature.join(
                (
                    target_data
                    .select(key_list + ['datetime', 'target'])
                    .with_columns(
                        pl.col("datetime") + pl.duration(days=day_lag)
                    )
                    .rename({"target": f"target_lag_{day_lag}"})
                ),
                on = key_list + ['datetime'], how='left'
            )

            target_feature = target_feature.join(
                aggregation_by_date.with_columns(
                    pl.col("date") + pl.duration(days=day_lag)
                ).rename({"target": f"avg_day_target_lag_{day_lag}"}),
                on = key_list + ['date'], how='left'
            )
            
            for col in key_list:
                target_feature = target_feature.join(
                    aggregation_by_dict[col].with_columns(
                        pl.col("datetime") + pl.duration(days=day_lag)
                    ).rename({"target": f"avg_{col}_target_lag_{day_lag}"}),
                    on = join_by_dict[col] + ['datetime'], how='left'
                )

        target_lag_for_stats = [f"target_lag_{day_lag}" for day_lag in range(2, self.target_n_lags+1)]
        target_feature = target_feature.filter(
            (
                pl.any_horizontal(
                    (
                        pl.col(col).is_null() 
                        for col in target_lag_for_stats
                    )
                ).not_()
            )
        )

        # #add mean lag over target
        target_feature = (
            target_feature
            #all aggregation
            .with_columns(
                #target lag mean
                (
                    pl.concat_list(pl.col(target_lag_for_stats))
                    .list.mean().cast(pl.Float32).alias('target_mean_all_lag')
                ),
                (
                    pl.concat_list(pl.col(target_lag_for_stats[:4]))
                    .list.mean().cast(pl.Float32).alias('target_mean_all_lag_2_4')
                ),
                #target lag min
                (
                    pl.concat_list(pl.col(target_lag_for_stats))
                    .list.min().cast(pl.Float32).alias('target_min_all_lag')
                ),
                (
                    pl.concat_list(pl.col(target_lag_for_stats[:4]))
                    .list.min().cast(pl.Float32).alias('target_min_all_lag_2_4')
                ),
                #target lag max
                (
                    pl.concat_list(pl.col(target_lag_for_stats))
                    .list.max().cast(pl.Float32).alias('target_max_all_lag')
                ),
                (
                    pl.concat_list(pl.col(target_lag_for_stats[:4]))
                    .list.max().cast(pl.Float32).alias('target_max_all_lag_2_4')
                )
            )
            #vs all aggregation
            .with_columns(
                (
                    (pl.col('target_lag_2')/(1+pl.col('target_mean_all_lag')))
                    .cast(pl.Float32).alias('target_lag_2_vs_mean_all')
                ),
                (
                    (pl.col('target_lag_2')/(1+pl.col('target_min_all_lag')))
                    .cast(pl.Float32).alias('target_lag_2_vs_min_all')
                ),
                (
                    (pl.col('target_lag_2')/(1+pl.col('target_max_all_lag')))
                    .cast(pl.Float32).alias('target_lag_2_vs_max_all')
                )
            )
        )
        
        self.target_data = target_feature.drop(
            ['target', 'date']
        )

    def create_train_feature(self) -> None:
        self.main_data = self.main_data.with_columns(
            pl.col('datetime').dt.date().cast(pl.Date).alias('date'),
            pl.col('datetime').dt.year().cast(pl.UInt16).alias('year'),
            pl.col('datetime').dt.quarter().cast(pl.UInt8).alias('quarter'),
            pl.col('datetime').dt.month().cast(pl.UInt8).alias('month'),
            pl.col('datetime').dt.week().cast(pl.UInt8).alias('week'),
            pl.col('datetime').dt.hour().cast(pl.UInt8).alias('hour'),
            pl.col('datetime').dt.day().cast(pl.UInt16).alias('day'),
            
            pl.col('datetime').dt.ordinal_day().cast(pl.UInt16).alias('ordinal_day'),
            pl.col('datetime').dt.weekday().cast(pl.UInt8).alias('weekday'),
            
        )
        self.main_data = self.main_data.with_columns(
            (np.pi * pl.col("ordinal_day") / 183).sin().cast(pl.Float32).alias("sin(ordinal_day)"),
            (np.pi * pl.col("ordinal_day") / 183).cos().cast(pl.Float32).alias("cos(ordinal_day)"),
            (np.pi * pl.col("hour") / 12).sin().cast(pl.Float32).alias("sin(hour)"),
            (np.pi * pl.col("hour") / 12).cos().cast(pl.Float32).alias("cos(hour)")
        )
        
        if not self.inference:
            #create fold time col -> incremental index over dateself.fold_time_col
            index_date_dict =  {
                row_['date']: i
                for i, row_ in (
                    self.main_data.select(
                        pl.col('date').unique().sort()
                        .dt.to_string(format="%Y/%m/%d")
                    )
                    .collect().to_pandas().iterrows()
                )
            }

            self.main_data = self.main_data.with_columns(
                pl.col('date').dt.to_string(format="%Y/%m/%d")
                .map_dict(index_date_dict)
                .alias(self.fold_time_col)
                .cast(pl.UInt16)
            )

        #capture every holiday
        min_year = self._collect_item_utils(
            self.main_data.select('year').min()
        )
        max_year = self._collect_item_utils(
            self.main_data.select('year').max()
        )
                
        estonian_holidays = list(
            holidays.country_holidays('EE', years=range(min_year-1, max_year+1)).keys()
        )

        # add holiday as a dummy 0, 1 variable
        self.main_data = self.main_data.with_columns(
            pl.when(
                pl.col('date')
                .is_in(estonian_holidays)
            ).then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.UInt8).alias('holiday')
        )
        
    def add_additional_feature(self) -> None:
        if not self.inference:
            n_rows_begin = self._collect_item_utils(
                self.data.select(pl.count())
            )
        
        #production target ~ installed_capacity * surface_solar_radiation_downwards / (temperature + 273.15)
        #https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/468654
        production_capacity_temperature_operator = [
                (
                    (
                        pl.col('installed_capacity') * 
                        pl.col(f'surface_solar_radiation_downwards_hours_ahead_{hour}')
                    ) /
                    (pl.col(f'temperature_hours_ahead_{hour}') + pl.lit(273.15, dtype=pl.Float32))
                ).alias(f'production_capacity_temperature_{hour}')
                for hour in range(22, 46)
        ]
        #same but on installed_capacity log1p
        production_capacitylog1p_temperature_operator = [
                (
                    (
                        pl.col('installed_capacity_log1p') * 
                        pl.col(f'surface_solar_radiation_downwards_hours_ahead_{hour}')
                    ) /
                    (pl.col(f'temperature_hours_ahead_{hour}') + pl.lit(273.15, dtype=pl.Float32))
                ).alias(f'production_capacity_log1p_temperature_{hour}')
                for hour in range(22, 46)
        ]
        self.data = self.data.with_columns(
            production_capacity_temperature_operator +
            production_capacitylog1p_temperature_operator
        )
        
        if not self.inference:
            n_rows_end = self._collect_item_utils(
                self.data.select(pl.count())
            )

            assert n_rows_begin == n_rows_end
    
    def merge_all(self) -> None:      
        if not self.inference:
      
            n_rows_begin = self._collect_item_utils(
                self.main_data.select(pl.count())
            )

        #Merge all datasets
        #merge with client
        self.data = self.main_data.join(
            self.client_data, how='left', 
            on=['county', 'is_business', 'product_type', 'date']
        )

        #merge with electricity
        self.data = self.data.join(
            self.electricity_data, how='left',
            on=['datetime'],
        )
        
        #merge with gas
        self.data = self.data.join(
            self.gas_data, how='left',
            on=['date']
        )
        
        #merge with weather
        self.data = self.data.join(
            self.forecast_weather_data, how='left',
            left_on = ['date', 'county'],
            right_on=  ['date', 'county']
        )

        #merge with historical
        self.data = self.data.join(
            self.historical_weather_data, how='left',
            on = ['date', 'county'],
        )
        self.data = self.data.join(
            self.target_data, how='left',
            on = ['datetime', 'county', 'is_business', 'product_type', 'is_consumption'],
        )

        if not self.inference:
            n_rows_end = self._collect_item_utils(
                self.data.select(pl.count())
            )

            assert n_rows_begin == n_rows_end