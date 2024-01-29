import os
import json
import pickle

import polars as pl
import lightgbm as lgb

from typing import Any, Union

class LgbmInit():
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], inference_setup: str=None,
            log_evaluation:int =1, fold_name: str = 'fold_info'
        ):
        if inference_setup is None:
            self.inference_setup = 'blend'
        else:
            if inference_setup not in ['single', 'blend']:
                raise ValueError
            else:
                self.inference_setup = inference_setup
        
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
        
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name
        )
        self.metric_eval: str = metric_eval
        self.n_fold: int = config_dict['N_FOLD']
        
        self.target_col_name: str = config_dict['TARGET_COL']
        self.fold_name: str = fold_name
        self.useless_col_list: list[str] = [
            'datetime', 'date', 'date_order_kfold',
            'data_block_id', 'prediction_unit_id',
            'row_id', 'current_fold', 'year'
        ]
        self.categorical_col_list: list[str] = [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
        ]
        self.log_evaluation: int = log_evaluation
        self.data: pl.LazyFrame = None
        self.params_lgb: dict[str, Any] = params_lgb
        
        self.model_all_data: lgb.Booster = None
        self.model_list: list[lgb.Booster] = []
        self.progress_list: list = []
        self.best_result: dict[str, Union[int, float]] = None
        
        self.feature_list: list[str] = []
        self.base_feature: list[str] = [
            'product_type',
            'county',
            'eic_count',
            'installed_capacity',
            'is_business',
            'target',
            'lowest_price_per_mwh',
            'highest_price_per_mwh',
            'euros_per_mwh',
            'temperature_hours_ahead_',
            'dewpoint_hours_ahead_',
            'cloudcover_high_hours_ahead_',
            'cloudcover_low_hours_ahead_',
            'cloudcover_mid_hours_ahead_',
            'cloudcover_total_hours_ahead_',
            '10_metre_u_wind_component_hours_ahead_',
            '10_metre_v_wind_component_hours_ahead_',
            'direct_solar_radiation_hours_ahead_',
            'surface_solar_radiation_downwards_hours_ahead_',
            'snowfall_hours_ahead_',
            'total_precipitation_hours_ahead_',
            'temperature_hours_ago_',
            'dewpoint_hours_ago_',
            'rain_hours_ago_',
            'snowfall_hours_ago_',
            'surface_pressure_hours_ago_',
            'cloudcover_total_hours_ago_',
            'cloudcover_low_hours_ago_',
            'cloudcover_mid_hours_ago_',
            'cloudcover_high_hours_ago_',
            'windspeed_10m_hours_ago_',
            'winddirection_10m_hours_ago_',
            'shortwave_radiation_hours_ago_',
            'direct_solar_radiation_hours_ago_',
            'diffuse_radiation_hours_ago_',
        ]
    def create_experiment_structure(self) -> None:
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)

    def load_model(self) -> None: 
        self.load_used_feature()
        self.load_best_result()
        self.load_params()
        self.load_model_list()
        # self.load_model_all_data()
        
    def save_progress_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.progress_list, file)

    def load_progress_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'progress_list_lgb.pkl'
            ), 'rb'
        ) as file:
            self.progress_list = pickle.load(file)

    def save_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'w'
        ) as file:
            json.dump(self.params_lgb, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'r'
        ) as file:
            self.params_lgb = json.load(file)
    
    def save_best_result(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'best_result_lgb.txt'
            ), 'w'
        ) as file:
            json.dump(self.best_result, file)
        
    def load_best_result(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'best_result_lgb.txt'
            ), 'r'
        ) as file:
            self.best_result = json.load(file)
            
    def save_pickle_model_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.model_list, file)
    
    def load_pickle_model_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'rb'
        ) as file:
            self.model_list = pickle.load(file)

    def load_model_all_data(self) -> None:
        self.model_all_data = lgb.Booster(
            params=self.params_lgb,
            model_file=os.path.join(
                self.experiment_path,
                f'lgb_all.txt'
            )
        )
    def load_model_list(self) -> None:
        
        self.model_list = [
            lgb.Booster(
                params=self.params_lgb,
                model_file=os.path.join(
                    self.experiment_path,
                    f'lgb_{fold_}.txt'
                )
            )
            for fold_ in range(self.n_fold)
        ]    
           
    def save_used_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'feature_model': self.feature_list
                }, 
                file
            )
            
    def load_used_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'r'
        ) as file:
            self.feature_list = json.load(file)['feature_model']