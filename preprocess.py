if __name__=='__main__':
    from src.utils.import_utils import import_config
    from src.preprocess import PreprocessPipeline

    config_dict = import_config()
    
    enefit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        target_n_lags=14, agg_target_n_lags=[2, 3, 7, 14], 
        embarko_skip=60
    )
    enefit_preprocessor()