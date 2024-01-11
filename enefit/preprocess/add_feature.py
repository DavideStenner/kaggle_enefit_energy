import polars as pl
import holidays

from enefit.preprocess.initialization import EnefitInit

class EnefitFeature(EnefitInit):
    
    def copy_starting_dataset(self) -> None:
        #keep starting dataset without modification -> used during inference to update it
        self.client_data = self.starting_client_data
        self.gas_data = self.starting_gas_data
        self.electricity_data = self.starting_electricity_data
        self.forecast_weather_data = self.starting_forecast_weather_data
        self.historical_weather_data = self.starting_historical_weather_data
        self.train_data = self.starting_train_data
        
        
    def create_client_feature(self) -> None:
        #add new column
        self.client_data = self.client_data.with_columns(
            #clean date column
            (pl.col("date") + pl.duration(days=2)).cast(pl.Date),
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
        
        
        #pivot everything to get future weather data -> collect not working for pivot 
        self.forecast_weather_data = self.forecast_weather_data.collect().pivot(
            values=training_variable, 
            index=index_variable, columns='hours_ahead'
        ).lazy()
        
    def create_historical_weather_feature(self) -> None:
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
        )
        number_null_join = self.historical_weather_data.select('county').null_count()
    
        assert number_null_join.collect().item() == 0
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
        
        #pivot everything to get future weather data -> collect not working for pivot 
        self.historical_weather_data = self.historical_weather_data.collect().pivot(
            values=training_variable, 
            index=index_variable, columns='hours_ago'
        ).lazy()

    def create_train_feature(self) -> None:
        original_num_rows = self.train_data.select(pl.count()).collect().item()
        
        grouped_col_list: list[str] = ['date', 'prediction_unit_id', 'is_consumption']
        agg_by_list: list[str] = ['county', 'is_business', 'product_type']
        
        self.train_data = self.train_data.with_columns(
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
        
        #create fold time col -> incremental index over dateself.fold_time_col
        index_date_dict =  {
            row_['date']: i
            for i, row_ in (
                self.train_data.select(
                    pl.col('date').unique().sort()
                    .dt.to_string(format="%Y/%m/%d")
                )
                .collect().to_pandas().iterrows()
            )
        }

        self.train_data = self.train_data.with_columns(
            pl.col('date').dt.to_string(format="%Y/%m/%d")
            .map_dict(index_date_dict)
            .alias(self.fold_time_col)
            .cast(pl.UInt16)
        )

        #capture every holiday
        min_year = self.train_data.select('year').min().collect().item()
        max_year = self.train_data.select('year').max().collect().item()
        
        #get min block day, max block day to create full dataset
        min_date = self.train_data.select('date').min().collect().item()
        max_date = self.train_data.select('date').max().collect().item()
        
        estonian_holidays = list(
            holidays.country_holidays('EE', years=range(min_year-1, max_year+1)).keys()
        )

        #add holiday as a dummy 0, 1 variable
        self.train_data = self.train_data.with_columns(
            pl.when(
                pl.col('date')
                .is_in(estonian_holidays)
            ).then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.UInt8).alias('holiday')
        )

        #calculate target -> date, prediction_unit_id, is_consumption is row key
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
            #ensure that every data block id is inserted -> no lag error
            pl.LazyFrame(
                pl.date_range(min_date, max_date, interval='1d', eager=True)
                .cast(pl.Date)
                .alias('date')
            )
            .join(
                self.train_data.select('prediction_unit_id').unique(),
                how = 'cross'
            )
            .join(
                self.train_data.select('is_consumption').unique(),
                how='cross'
            )
            #add detail over county, is_business, product_type used to join aggregation
            #to drop before joining with train data
            .join(
                self.train_data.select(['prediction_unit_id'] + agg_by_list).unique(),
                how='left', on='prediction_unit_id'
            )
        ).sort(['is_consumption', 'prediction_unit_id', 'date'], descending=False)
        
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
            join_col_list = ['date', 'is_consumption'] + filter_col_for_agg
            
            #add target averaged over ...
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

        #merge with gas
        self.data = self.data.join(
            self.historical_weather_data, how='left',
            on = ['date', 'county'],
        )
        
        n_rows_end = self.data.select(pl.count()).collect().item()

        assert n_rows_begin == n_rows_end