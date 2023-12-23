if __name__=='__main__':
    from enefit.utils.import_utils import import_config, import_params
    from enefit.model.lgbm import ModelPipeline

    config_dict = import_config()
    params_model = import_params()
    
    trainer = ModelPipeline(
        experiment_name='test',
        params_lgb=params_model,
        config_dict=config_dict
    )
    trainer.train()