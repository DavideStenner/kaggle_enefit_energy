if __name__=='__main__':
    from src.utils.import_utils import import_config, import_params
    from src.model.lgbm import ModelPipeline

    config_dict = import_config()
    params_model, experiment_name = import_params(model='lgb')
    
    trainer = ModelPipeline(
        experiment_name=experiment_name,
        params_lgb=params_model,
        config_dict=config_dict,
        metric_eval='l1', log_evaluation=25
    )
    trainer.train_explain()