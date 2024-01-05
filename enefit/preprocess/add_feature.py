import polars as pl
from enefit.preprocess.initialization import EnefitInit

class EnefitFeature(EnefitInit):
    
    def create_client_feature(self) -> None:
        #add new column
        self.client_data = self.client_data.with_columns(
            (
                pl.col('eic_count').mean()
                .over(['product_type', 'county', 'is_business'])
                .cast(pl.UInt16)
                .alias('eic_count_mean')
            ),
            (
                pl.col('installed_capacity').mean()
                .over(['product_type', 'county', 'is_business'])
                .cast(pl.Float32)
                .alias('installed_capacity_mean')
            )
        )
        #data not needed during merging
        self.client_data = self.client_data.drop(['date'])
        
    def create_gas_feature(self) -> None:
        self.gas_data = self.gas_data.with_columns(
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
        self.gas_data = self.gas_data.drop(['origin_date', 'forecast_date'])
        
    def create_electricity_feature(self) -> None:
        #not needed during merge
        self.electricity_data = self.electricity_data.drop('origin_date')
            
    def create_forecast_weather_feature(self) -> None:
        
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
            'county', 'origin_datetime', 'data_block_id'
        ]
    
        #add county
        self.forecast_weather_data = self.forecast_weather_data.join(
            self.location_data, on=['latitude', 'longitude']
        )
        number_null_join = self.forecast_weather_data.select('county').null_count()
    
        assert number_null_join.collect().item() == 0
        
        original_col_dict = {
            col: self.forecast_weather_data.select(col).dtypes[0]
            for col in training_variable
        }
        #aggregate over county
        self.forecast_weather_data = (
            self.forecast_weather_data.group_by(index_variable + ['hours_ahead'])
            .agg(
                [
                    pl.col(col).mean().cast(original_col_dict[col]) 
                    for col in training_variable
                ]
            )
        )
        
        
        #pivot everything to get future weather data
        self.forecast_weather_data = self.forecast_weather_data.collect().pivot(
            values=training_variable, 
            index=index_variable, columns='hours_ahead'
        ).lazy()
        
    def create_historical_weather_feature(self) -> None:
        training_variable: list[str] = [
            'temperature', 'dewpoint', 'rain',
            'snowfall', 'surface_pressure', 'cloudcover_total',
            'cloudcover_low', 'cloudcover_mid',
            'cloudcover_high', 'windspeed_10m',
            'winddirection_10m', 'shortwave_radiation',
            'direct_solar_radiation', 'diffuse_radiation'
        ]
        index_variable: list[str] = [
            'county', 'data_block_id'
        ]
        
        #add county
        self.historical_weather_data = self.historical_weather_data.join(
            self.location_data, on=['latitude', 'longitude']
        )
        number_null_join = self.historical_weather_data.select('county').null_count()
    
        assert number_null_join.collect().item() == 0
        self.historical_weather_data = (
            self.historical_weather_data
            .sort(
                ['latitude', 'longitude', 'data_block_id', 'datetime'],
                descending=[False, False, False, True]
            )
            .with_columns(
                pl.arange(0, pl.count())
                .over(['latitude', 'longitude', 'data_block_id'])
                .cast(pl.UInt8).alias('hours_ago')
            )
        )
        
        original_col_dict = {
            col: self.historical_weather_data.select(col).dtypes[0]
            for col in training_variable
        }
        #aggregate over county
        self.historical_weather_data = (
            self.historical_weather_data.group_by(index_variable + ['hours_ago'])
            .agg(
                [
                    pl.col(col).mean().cast(original_col_dict[col]) 
                    for col in training_variable
                ]
            )
        )

        self.historical_weather_data = self.historical_weather_data.collect().pivot(
            values=training_variable, 
            index=index_variable, columns='hours_ago'
        ).lazy()

    def create_train_feature(self) -> None:
        original_num_rows = self.train_data.select(pl.count()).collect().item()
        
        grouped_col_list: list[str] = ['data_block_id', 'prediction_unit_id', 'is_consumption']
        agg_by_list: list[str] = ['county', 'is_business', 'product_type']
        
        self.train_data = self.train_data.with_columns(
            pl.col('datetime').dt.year().cast(pl.UInt16).alias('year'),
            pl.col('datetime').dt.quarter().cast(pl.UInt8).alias('quarter'),
            pl.col('datetime').dt.month().cast(pl.UInt8).alias('month'),
            pl.col('datetime').dt.week().cast(pl.UInt8).alias('week'),
            pl.col('datetime').dt.hour().cast(pl.UInt8).alias('hour'),
            pl.col('datetime').dt.day().cast(pl.UInt16).alias('day'),
            
            pl.col('datetime').dt.ordinal_day().cast(pl.UInt16).alias('ordinal_day'),
            pl.col('datetime').dt.weekday().cast(pl.UInt8).alias('weekday'),
        )
        
        #calculate target -> data_block_id, prediction_unit_id, is_consumption is row key
        #useless aggregation used only to ensure no duplicates
        revealed_targets_avg = (
            self.train_data.select(
                grouped_col_list + ['target']
            ).group_by(grouped_col_list)
            .agg(
                pl.col('target').mean().cast(pl.Float32)
            )
        )
        #create full dataset so shift is correct
        full_revealed_targets = (
            revealed_targets_avg.select('data_block_id').unique()
            .join(
                revealed_targets_avg.select('prediction_unit_id').unique(),
                how = 'cross'
            )
            .join(
                revealed_targets_avg.select('is_consumption').unique(),
                how='cross'
            )
            #add detail over county, is_business, product_type used to join aggregation
            #to drop before joining with train data
            .join(
                self.train_data.select(['prediction_unit_id'] + agg_by_list).unique(),
                how='left', on='prediction_unit_id'
            )
        ).sort(grouped_col_list[::-1], descending=False)
        
        full_revealed_targets = full_revealed_targets.join(
            revealed_targets_avg,
            on=grouped_col_list,
            how='left'
        )
        #add aggregation by multiple col with lag
        #add target shifted
        list_agg_operation_shift = [
            (
                pl.col('target').shift(lag_).over(['prediction_unit_id', 'is_consumption'])
                .cast(pl.Float32).alias(f'target_lag_{lag_}')
            )
            for lag_ in range(2, self.target_n_lags+1)
        ]

        for col in agg_by_list:
            filter_col_for_agg = [filter_col for filter_col in agg_by_list if filter_col!=col]
            
            #add date and is consumption to calculate aggregation by col
            join_col_list = ['data_block_id', 'is_consumption'] + filter_col_for_agg
            
            agg_revealed_targets_avg = (
                self.train_data.select(
                    join_col_list + ['target']
                ).group_by(join_col_list)
                .agg(
                    pl.col('target').mean().cast(pl.Float32).alias(f'avg_target_{col}')
                )
            )

            full_revealed_targets = full_revealed_targets.join(
                agg_revealed_targets_avg,
                on=join_col_list,
                how='left'
            )
            #add multiple avg target operation
            list_agg_operation_shift += [
                (
                    pl.col(f'avg_target_{col}').shift(lag_).over(['prediction_unit_id', 'is_consumption'])
                    .cast(pl.Float32).alias(f'avg_target_{col}_lag_{lag_}')
                )
                for lag_ in self.agg_target_n_lags
            ]

        #add multiple target now
        full_revealed_targets = full_revealed_targets.with_columns(
            list_agg_operation_shift
        ).drop(['target'] + [f'avg_target_{drop_col}' for drop_col in agg_by_list] + agg_by_list)
        
        self.train_data = self.train_data.join(
            full_revealed_targets, 
            how='left', 
            left_on = grouped_col_list,
            right_on = grouped_col_list
        )
        final_num_rows = self.train_data.select(pl.count()).collect().item()
        assert final_num_rows == original_num_rows
        
    def merge_all(self) -> None:
        n_rows_begin = self.train_data.select(pl.count()).collect().item()

        #Merge all datasets
        
        #merge with client
        self.data = self.train_data.join(
            self.client_data, how='left', 
            on=['county', 'is_business', 'product_type', 'data_block_id']
        )
                
        #merge with electricity
        self.data = self.data.join(
            self.electricity_data, how='left',
            left_on=['datetime', 'data_block_id'],
            right_on=['forecast_date', 'data_block_id']
        )
        
        #merge with gas
        self.data = self.data.join(
            self.gas_data, how='left',
            on=['data_block_id']
        )
        
        #merge with weather
        self.data = self.data.join(
            self.forecast_weather_data, how='left',
            left_on = ['datetime', 'data_block_id', 'county'],
            right_on=  ['origin_datetime', 'data_block_id', 'county']
        )

        #merge with gas
        self.data = self.data.join(
            self.historical_weather_data, how='left',
            on = ['data_block_id', 'county'],
        )
        
        n_rows_end = self.data.select(pl.count()).collect().item()

        assert n_rows_begin == n_rows_end