from typing import Any

class EnefitInit():
    def __init__(self, 
            config_dict: dict[str, Any], target_n_lags: int, 
            agg_target_n_lags: list[int] = [2, 3, 7, 15], embarko_skip: int=60
        ):
        
        self.target_n_lags: int = target_n_lags
        self.agg_target_n_lags: int = agg_target_n_lags
        self.path_original_data: str = config_dict['PATH_ORIGINAL_DATA']
        self.config_dict: dict[str, Any] = config_dict
        self.embarko_skip: int = embarko_skip
        self.n_folds: int = config_dict['N_FOLD']
