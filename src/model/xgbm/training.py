import os
import gc
import polars as pl
import xgboost as xgb

from src.model.xgbm.initialization import XgbInit

class XgbTrainer(XgbInit):
    def _init_train(self) -> None:
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
        if self.importance_feature_list is None:
            print('Using all feature')
            self.feature_list = [
                col for col in data.columns
                if col not in self.useless_col_list + [self.fold_name, self.target_col_name]
            ]
        else:
            print(f'Using top {len(self.importance_feature_list)} feature')
            self.feature_list = (
                self.importance_feature_list + 
                [cat_col for cat_col in self.categorical_col_list if cat_col not in self.importance_feature_list]
            )
        
        #save feature list locally for later
        self.save_used_feature()
            
    def access_fold(self, fold_: int) -> pl.LazyFrame:
        fold_data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
        fold_data = fold_data.with_columns(
            (
                pl.col('fold_info').str.split(', ')
                .list.get(fold_).alias('current_fold')
            )
        )
        return fold_data

    def train(self) -> None:
        
        self._init_train()
        
        for fold_ in range(self.n_fold):
            print(f'\n\nStarting fold {fold_}\n\n\n')
            
            progress = {}

            print('Collecting dataset')
            fold_data = self.access_fold(fold_=fold_)
            
            train_filtered = fold_data.filter(
                (pl.col('current_fold') == 't') &
                (pl.col('target').is_not_null())
            )
            test_filtered = fold_data.filter(
                (pl.col('current_fold') == 'v') &
                (pl.col('target').is_not_null())
            )
            
            assert len(
                set(
                    train_filtered.select('date').collect().to_series().to_list()
                ).intersection(
                    test_filtered.select('date').collect().to_series().to_list()
                )
            ) == 0
            
            train_rows = train_filtered.select(pl.count()).collect().item()
            test_rows = test_filtered.select(pl.count()).collect().item()
            
            print(f'{train_rows} train rows; {test_rows} test rows; {len(self.feature_list)} feature')
            
            assert self.target_col_name not in self.feature_list
            
            train_matrix = xgb.DMatrix(
                train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                train_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float64').reshape((-1)),
                feature_names=self.feature_list
            )
            
            test_matrix = xgb.DMatrix(
                test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                test_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float64').reshape((-1)),
                feature_names=self.feature_list
            )

            print('Start training')
            model = xgb.train(
                params=self.params_xgb,
                dtrain=train_matrix, 
                num_boost_round=self.params_xgb['n_round'],
                evals=[(test_matrix, 'valid')],
                evals_result=progress, verbose_eval=self.log_evaluation
            )

            model.save_model(
                os.path.join(
                    self.experiment_path,
                    f'xgb_{fold_}.json'
                )
            )

            self.model_list.append(model)
            self.progress_list.append(progress)

            del train_matrix, test_matrix
            
            _ = gc.collect()

    def save_model(self)->None:
        self.save_pickle_model_list()
        self.save_params()
        self.save_progress_list()
            
