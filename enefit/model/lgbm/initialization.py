import os
import polars as pl
import lightgbm as lgb

from typing import Any, Union

class LgbmInit():
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], 
            log_evaluation:int =1, fold_name: str = 'fold_info'
        ):
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
            'row_id', 'current_fold'
        ]
        self.log_evaluation: int = log_evaluation
        self.data: pl.LazyFrame = None
        self.params_lgb: dict[str, Any] = params_lgb
        
        self.model_list: list[lgb.Booster] = []
        self.progress_list: list = []
        self.best_result: dict[str, Union[int, float]] = None
        
        self.feature_list: list[str] = []
        
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
        
