from typing import Any

from src.model.xgbm.training import XgbTrainer
from src.model.xgbm.initialization import XgbInit
from src.model.xgbm.inference import XgbInference
from src.model.xgbm.explainer import XgbExplainer

class XgbmPipeline(XgbTrainer, XgbExplainer, XgbInference):
    def __init__(self, 
            experiment_name:str, 
            params_xgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], inference_setup: str=None,
            log_evaluation:int =1, fold_name: str = 'fold_info', use_importance_filter: bool = False
        ):
        XgbInit.__init__(
            self, experiment_name=experiment_name, params_xgb=params_xgb,
            metric_eval=metric_eval, config_dict=config_dict, inference_setup=inference_setup,
            log_evaluation=log_evaluation, fold_name=fold_name, use_importance_filter=use_importance_filter
        )

    def activate_inference(self) -> None:
        self.load_model()
        self.inference = True
        
    def run_train(self) -> None:
        self.train()
        self.save_model()
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.run_train()
        self.explain_model()
