from typing import Any

from enefit.model.lgbm.training import LgbmTrainer
from enefit.model.lgbm.explainer import LgbmExplainer
from enefit.model.lgbm.initialization import LgbmInit

class LgbmPipeline(LgbmTrainer, LgbmExplainer):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], 
            log_evaluation:int =1, fold_name: str = 'fold_info'
        ):
        LgbmInit.__init__(
            self, experiment_name=experiment_name, params_lgb=params_lgb,
            metric_eval=metric_eval, config_dict=config_dict,
            log_evaluation=log_evaluation, fold_name=fold_name
        )

    def run_train(self) -> None:
        self.train()
        self.save_model()
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()
        
    def __call__(self) -> None:
        self.run_train()
        self.explain_model()