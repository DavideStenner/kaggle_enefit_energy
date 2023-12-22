if __name__=='__main__':
    from enefit.utils.import_utils import import_config
    from enefit.preprocess import PreprocessPipeline

    config_dict = import_config()
    
    enefit_preprocessor = PreprocessPipeline(config_dict=config_dict, target_n_lags=15)
    enefit_preprocessor()