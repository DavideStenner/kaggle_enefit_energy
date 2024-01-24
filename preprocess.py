if __name__=='__main__':
    from src.utils.import_utils import import_config
    from src.preprocess import PreprocessPipeline

    config_dict = import_config()
    
    enefit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        target_n_lags=2, agg_target_n_lags=[2], 
        embarko_skip=0
    )
    enefit_preprocessor()