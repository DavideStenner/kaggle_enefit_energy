import os
import gc
import json
import pickle
import polars as pl
import lightgbm as lgb

from typing import Any

class LgbmTrainer():
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], 
            log_evaluation:int =1, n_fold: int=5
        ):
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name
        )
        self.metric_eval: str = metric_eval
        self.n_fold: int = n_fold
        
        self.target_col_name: str = 'target'
        self.fold_name: str = 'fold_info'
        self.useless_col_list: list[str] = [
            'datetime',
            'data_block_id',
            'row_id', 'current_fold'
        ]
        self.log_evaluation: int = log_evaluation
        self.data: pl.LazyFrame = pl.scan_parquet(
            os.path.join(
                config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
        self.params_lgb: dict[str, Any] = params_lgb
        
        self.model_list: list = []
        self.progress_list: list = []

        self.feature_list = [
            col for col in self.data.columns
            if col not in self.useless_col_list + [self.fold_name, self.target_col_name]
        ]
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
        
    def access_fold(self, fold_: int) -> pl.LazyFrame:
        fold_data = self.data
        fold_data = fold_data.with_columns(
            (
                pl.col('fold_info').str.split(', ')
                .list.get(fold_).alias('current_fold')
            )
        )
        return fold_data

    def train(self) -> None:
        
        for fold_ in range(self.n_fold):
            print(f'\n\nStarting fold {fold_}\n\n\n')
            
            progress = {}

            callbacks_list = [
                lgb.record_evaluation(progress),
                lgb.log_evaluation(
                    period=self.log_evaluation, 
                    show_stdv=False
                )
            ]
            print('Collecting dataset')
            fold_data = self.access_fold(fold_=fold_)
            
            train_filtered = fold_data.filter(
                pl.col('current_fold') == 't'
            )
            test_filtered = fold_data.filter(
                pl.col('current_fold') == 'v'
            )
            
            train_matrix = lgb.Dataset(
                train_filtered.select(self.feature_list).collect().to_numpy().astype('float32'),
                train_filtered.select(self.target_col_name).collect().to_numpy().reshape((-1)).astype('float32')
            )
            
            test_matrix = lgb.Dataset(
                test_filtered.select(self.feature_list).collect().to_numpy().astype('float32'),
                test_filtered.select(self.target_col_name).collect().to_numpy().reshape((-1)).astype('float32')
            )

            print('Start training')
            model = lgb.train(
                params=self.params_lgb,
                train_set=train_matrix, 
                num_boost_round=self.params_lgb['n_round'],
                valid_sets=[test_matrix],
                valid_names=['valid'],
                callbacks=callbacks_list,
            )

            model.save_model(
                os.path.join(
                    self.experiment_path,
                    f'lgb_{fold_}.txt'
                )
            )

            self.model_list.append(model)
            self.progress_list.append(progress)

            del train_matrix, test_matrix
            
            _ = gc.collect()
            self.save_model()

    def save_model(self)->None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'w'
        ) as file:
            json.dump(self.params_lgb, file)
            
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.model_list, file)

        with open(
            os.path.join(
                self.experiment_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.progress_list, file)