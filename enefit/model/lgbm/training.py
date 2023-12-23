import os
import gc
import json
import pickle
import polars as pl
import lightgbm as lgb

from enefit.model.lgbm.initialization import LgbmInit

class LgbmTrainer(LgbmInit):
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