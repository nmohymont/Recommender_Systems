import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from loaders import load_ratings
from constants import Constant as C

ratings = load_ratings(surprise_format=False, use_implicit=False)
reader  = Reader(rating_scale=C.RATINGS_SCALE)
data    = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
trainset, testset = train_test_split(data, test_size=0.20, random_state=1)

def objective(trial):
    params = {
        'n_factors': trial.suggest_categorical('n_factors', [20, 50, 75, 100, 150, 200]),
        'n_epochs':  trial.suggest_int('n_epochs', 20, 100),
        'lr_all':    trial.suggest_float('lr_all', 0.001, 0.02, log=True),
        'reg_all':   trial.suggest_float('reg_all', 0.01, 0.15, log=True),
        'random_state': 42,
    }
    model = SVD(**params)
    model.fit(trainset)
    preds = model.test(testset)
    return accuracy.rmse(preds, verbose=False)

print('n_trials,best_rmse,n_factors,n_epochs,lr_all,reg_all')
for n in [10, 20, 30, 50, 100]:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    p = study.best_params
    print(f"{n},{study.best_value:.4f},{p['n_factors']},{p['n_epochs']},{p['lr_all']:.6f},{p['reg_all']:.6f}")