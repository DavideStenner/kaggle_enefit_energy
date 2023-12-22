from typing import Any

class EnefitInit():
    def __init__(self, config_dict: dict[str, Any], target_n_lags: int):
        
        self.target_n_lags = target_n_lags
        self.path_original_data: str = config_dict['PATH_ORIGINAL_DATA']
        self.config_dict: dict[str, Any] = config_dict

