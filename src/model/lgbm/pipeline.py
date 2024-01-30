from typing import Any

from src.model.lgbm.training import LgbmTrainer
from src.model.lgbm.explainer import LgbmExplainer
from src.model.lgbm.initialization import LgbmInit
from src.model.lgbm.inference import LgbmInference

class LgbmPipeline(LgbmTrainer, LgbmExplainer, LgbmInference):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], inference_setup: str=None,
            log_evaluation:int =1, fold_name: str = 'fold_info', use_importance_filter: bool = False, number_importance_feature: int = None
        ):
        LgbmInit.__init__(
            self, experiment_name=experiment_name, params_lgb=params_lgb,
            metric_eval=metric_eval, config_dict=config_dict, inference_setup=inference_setup,
            log_evaluation=log_evaluation, fold_name=fold_name, use_importance_filter=use_importance_filter, number_importance_feature = number_importance_feature
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
        # self.all_data_train(num_round=self.best_result['best_epoch'])