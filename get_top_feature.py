if __name__=='__main__':
    import os
    import argparse
    import pandas as pd
    
    from src.utils.import_utils import import_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="final_lgb")
    
    args = parser.parse_args()
    
    config_dict = import_config()
    importance_ordered_feature_list: list[str] = pd.read_excel(
            os.path.join(
                config_dict['PATH_EXPERIMENT'],
                args.experiment, 
                'feature_importances.xlsx'
            )
        ).sort_values(by='average', ascending=False)['feature'].values.tolist()
    
    with open('config/best_feature.txt', "w") as file:
        # Write each element of the list to the file
        file.write('\n'.join(importance_ordered_feature_list))