# local imports
from models import *


class EvalConfig:
    
    models = [
        #("baseline_1", ModelBaseline1, {}), 
        #("baseline_2", ModelBaseline2, {}), 
        #("baseline_3", ModelBaseline3, {}),
        ("SVD", SVD, {"random_state": 1}),
        #("random_score", ContentBased, {"features_method":None, "regressor_method" :'random_score'}),
        #("random_sample", ContentBased, {"features_method":None, "regressor_method" :'random_sample'}),
        ("linear_regression", ContentBased, {"features_method":"title_length", "regressor_method" :'linear_regression'})
         # model_name, model class, model parameters (dict)
    ]
    split_metrics = ["rmse"] #"mae","rmse","mse","fcp"
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the number of recommendations (> 1) --
