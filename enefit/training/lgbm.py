import os
import gc
import json
import pickle
import pandas as pd
import polars as pl
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

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

    def evaluate_score(self) -> None:        
        # Find best epoch
        with open(
            os.path.join(
                self.experiment_path,
                'progress_list_lgb.pkl'
            ), 'rb'
        ) as file:
            progress_list = pickle.load(file)

        progress_dict = {
            'time': range(self.params_lgb['n_round']),
        }

        progress_dict.update(
                {
                    f"{self.metric_eval}_fold_{i}": progress_list[i]['valid'][self.metric_eval]
                    for i in range(self.n_fold)
                }
            )

        progress_df = pd.DataFrame(progress_dict)
        progress_df[f"average_{self.metric_eval}"] = progress_df.loc[
            :, [self.metric_eval in x for x in progress_df.columns]
        ].mean(axis =1)
        
        progress_df[f"std_{self.metric_eval}"] = progress_df.loc[
            :, [self.metric_eval in x for x in progress_df.columns]
        ].std(axis =1)

        best_epoch_lgb = int(progress_df[f"average_{self.metric_eval}"].argmin())
        best_score_lgb = progress_df.loc[
            best_epoch_lgb,
            f"average_{self.metric_eval}"
        ]
        lgb_std = progress_df.loc[
            best_epoch_lgb, f"std_{self.metric_eval}"
        ]

        print(f'Best epoch: {best_epoch_lgb}, CV-Auc: {best_score_lgb:.5f} ± {lgb_std:.5f}')

        self.best_result = {
            'best_epoch': best_epoch_lgb+1,
            'best_score': best_score_lgb
        }

        with open(
            os.path.join(
                self.experiment_path,
                'best_result_lgb.txt'
            ), 'w'
        ) as file:
            json.dump(self.best_result, file)
        
    def get_feature_importance(self) -> None:
   
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'rb'
        ) as file:
            model_list = pickle.load(file)

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(model_list):
            feature_importances[f'fold_{fold_}'] = model.feature_importance(
                importance_type='gain', iteration=self.best_result['best_epoch']
            )

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)

        fig = plt.figure(figsize=(12,8))
        sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature')
        plt.title(f"50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(self.experiment_path, 'importance_plot.png')
        )
        plt.close(fig)
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()