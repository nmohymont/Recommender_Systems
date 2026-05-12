# local imports
from models import *


class EvalConfig:
    
    models = [
        #("baseline_1", ModelBaseline1, {}), 
        #("baseline_2", ModelBaseline2, {}), 
        #("baseline_3", ModelBaseline3, {}),
        #("SVD", SVD, {"random_state": 1}),
        #("random_score", ContentBased, {"features_method":None, "regressor_method" :'random_score'}),
        #("random_sample", ContentBased, {"features_method":None, "regressor_method" :'random_sample'}),
        #("linear_regression", ContentBased, {"features_method":"title_length", "regressor_method" :'linear_regression'}),
        #("ridge_regression_log", ContentBased, {"features_method":'visuals_log', "regressor_method" :'ridge_regression'}),
        #("ridge_regression_quantile", ContentBased, {"features_method":'visuals_quantile', "regressor_method" :'ridge_regression'}),
        #("ridge_regression_quantilelog", ContentBased, {"features_method": 'visuals_quantilelog', "regressor_method" :'ridge_regression'}),
        ("random_forest_log", ContentBased, {"features_method": 'visuals_log', "regressor_method" :'random_forest_with_selection'}),
        ("random_forest_quantile", ContentBased, {"features_method": 'visuals_quantile', "regressor_method" :'random_forest_with_selection'}),
        ("random_forest-quantilelog", ContentBased, {"features_method": 'visuals_quantilelog', "regressor_method" :'random_forest_with_selection'}),
         
         # model_name, model class, model parameters (dict)
    ]
    split_metrics = ["rmse"] #"mae","rmse","mse","fcp"
    loo_metrics = [] #"hit_rate"
    full_metrics = [] #"novelty"

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the number of recommendations (> 1) --
