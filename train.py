if __name__=='__main__':
    from enefit.utils.import_utils import import_config
    from enefit.model.lgbm import ModelPipeline

    config_dict = import_config()
    trainer = ModelPipeline(
        experiment_name='test',
        params_lgb={
            "boosting_type": "gbdt",
            "objective": "mae",
            "n_jobs": -1,
            "num_leaves": 256,
            "learning_rate": 0.1,
            "feature_fraction": 0.75,
            "bagging_freq": 1,
            "bagging_fraction": 0.80,
            "lambda_l2": 1,
            "verbosity": -1,
            "n_round": 200
        },
        config_dict=config_dict
    )
    trainer.train()