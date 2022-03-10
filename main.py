import os
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from econml.dml import DML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner
from econml.grf import CausalForest
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LassoLarsCV, RidgeCV, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier
from models.data import IHDP, JOBS, NEWS, TWINS

class CausalForestWrapper(CausalForest):
    """ CausalForest doesn't work with GridSearchCV because CF.fit expects (X, T, Y) params,
    but GS passes (X, Y, T). This overwrite is to fix this.
    """
    def fit(self, X, Y, T, **kwargs):
        return super().fit(X, T, Y, **kwargs)

class RidgeCVClassifier(RidgeCV):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)

class LassoLarsCVClassifier(LassoLarsCV):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)

class KernelRidgeClassifier(KernelRidge):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)

def get_scaler(name):
    result = None
    if name == 'minmax':
        result = MinMaxScaler(feature_range=(-1, 1))
    elif name == 'std':
        result = StandardScaler()
    else:
        raise ValueError('Unknown scaler type.')
    return result

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dtype', type=str, choices=['ihdp', 'jobs', 'news', 'twins'])
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--sr', dest='save_results', action='store_true')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--scale_bin', action='store_true', default=False)
    parser.add_argument('--scale_y', action='store_true', default=False)
    parser.add_argument('--tbv', dest='transform_bin_vars', action='store_true')
    parser.add_argument('--ty', dest='transform_y', action='store_true')
    parser.add_argument('--cv', type=int, default=5)

    # Estimation
    parser.add_argument('--em', type=str, dest='estimation_model', choices=['dml', 'dr', 'xl', 'tl', 'cf', 'ridge', 'ridge-ipw', 'lasso', 'kr', 'kr-ipw', 'et', 'et-ipw', 'dt', 'dt-ipw', 'cb', 'cb-ipw', 'lgbm', 'lgbm-ipw', 'lr', 'lr-ipw', 'dummy'])
    parser.add_argument('--ebm', dest='estimation_base_model', type=str, choices=['lr', 'ridge', 'lasso', 'kr', 'et', 'dt', 'cb', 'lgbm'], default='lr')
    parser.add_argument('--ipw', dest='ipw_model', type=str, choices=['lr', 'kr', 'dt', 'et', 'cb', 'lgbm'], default='lr')
    parser.add_argument('--sfi', dest='save_features', action='store_true')

    return parser

def get_dataset(options):
    result = None
    num_scores = 3
    if options.dtype == 'ihdp':
        result = IHDP(options.data_path, options.iter)
    elif options.dtype == 'news':
        result = NEWS(options.data_path, options.iter)
    elif options.dtype == 'jobs':
        result = JOBS(options.data_path, options.iter)
        num_scores = 2
    elif options.dtype == 'twins':
        result = TWINS(options.data_path, options.iter)
    else:
        raise ValueError('Unknown dataset type selected.')
    return result, num_scores

def init_logger(options):
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(options.output_path, 'info.log'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def estimation_preproc(train, test):
    (x_tr, t_tr, y_tr), (x_test, t_test, y_test) = train, test

    ta_tr = np.zeros(shape=(x_tr.shape[0], 1))
    tb_tr = np.ones(shape=(x_tr.shape[0], 1))
    ta_test = np.zeros(shape=(x_test.shape[0], 1))
    tb_test = np.ones(shape=(x_test.shape[0], 1))    
    
    xta_tr = np.concatenate([x_tr, ta_tr], axis=1)
    xtb_tr = np.concatenate([x_tr, tb_tr], axis=1)
    xta_test = np.concatenate([x_test, ta_test], axis=1)
    xtb_test = np.concatenate([x_test, tb_test], axis=1)
    
    return (xta_tr, xtb_tr), (xta_test, xtb_test)

def _get_classifier(name, options):
    result = None
    if name in ('ridge', 'ridge-ipw'):
        result = RidgeCVClassifier(cv=options.cv)
    elif name == 'lr':
        result = LogisticRegressionCV(cv=options.cv, n_jobs=1, random_state=options.seed)
    elif name == 'lasso':
        result = LassoLarsCVClassifier(cv=options.cv, n_jobs=1)
    elif name in ('dt', 'dt-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(DecisionTreeClassifier(random_state=options.seed), param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('et', 'et-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(ExtraTreesClassifier(n_estimators=1000, bootstrap=True, random_state=options.seed, n_jobs=1), param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('kr', 'kr-ipw'):
        params = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
        result = GridSearchCV(KernelRidgeClassifier(), n_jobs=options.n_jobs, scoring="neg_mean_squared_error", param_grid=params, cv=options.cv)
    elif name in ('cb', 'cb-ipw'):
        params = {"depth": [6, 8, 10], "l2_leaf_reg": [1, 3, 10, 100]}
        result = GridSearchCV(CatBoostClassifier(iterations=1000, random_state=options.seed, verbose=False, thread_count=1), param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('lgbm', 'lgbm-ipw'):
        params = {"max_depth": [5, 7, 10], "reg_lambda": [0.1, 0, 1, 5, 10]}
        result = GridSearchCV(LGBMClassifier(n_estimators=1000, n_jobs=1, random_state=options.seed), param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    else:
        raise ValueError('Unknown classifier chosen.')
    return result

def _get_regressor(name, options):
    result = None
    if name == 'dummy':
        result = DummyRegressor()
    elif name in ('ridge', 'ridge-ipw'):
        result = RidgeCV(cv=options.cv)
    elif name == 'lr':
        result = LinearRegression(n_jobs=1)
    elif name == 'lasso':
        result = LassoLarsCV(cv=options.cv, n_jobs=1)
    elif name in ('dt', 'dt-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(DecisionTreeRegressor(random_state=options.seed), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('et', 'et-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(ExtraTreesRegressor(n_estimators=1000, bootstrap=True, random_state=options.seed, n_jobs=1), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('kr', 'kr-ipw'):
        params = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
        result = GridSearchCV(KernelRidge(), n_jobs=options.n_jobs, scoring="neg_mean_squared_error", param_grid=params, cv=options.cv)
    elif name in ('cb', 'cb-ipw'):
        params = {"depth": [6, 8, 10], "l2_leaf_reg": [1, 3, 10, 100]}
        result = GridSearchCV(CatBoostRegressor(iterations=1000, random_state=options.seed, verbose=False, thread_count=1), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('lgbm', 'lgbm-ipw'):
        params = {"max_depth": [5, 7, 10], "reg_lambda": [0.1, 0, 1, 5, 10]}
        result = GridSearchCV(LGBMRegressor(n_estimators=1000, n_jobs=1, random_state=options.seed), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=options.cv)
    else:
        raise ValueError('Unknown regressor chosen.')
    return result

def _get_model(options):
    result = None
    fit_type = 'econml'
    if options.estimation_model == 'dml':
        result = DML(model_y=_get_regressor(options.estimation_base_model, options), model_t=_get_classifier(options.estimation_base_model, options), model_final=_get_regressor(options.estimation_base_model, options), discrete_treatment=True, random_state=options.seed, fit_cate_intercept=True, cv=options.cv)
    elif options.estimation_model == 'dr':
        result = DRLearner(model_propensity=_get_classifier(options.estimation_base_model, options), model_regression=_get_regressor(options.estimation_base_model, options), model_final=_get_regressor(options.estimation_base_model, options), random_state=options.seed, cv=options.cv)
    elif options.estimation_model == 'xl':
        result = XLearner(models=_get_regressor(options.estimation_base_model, options), propensity_model=_get_classifier(options.estimation_base_model, options))
    elif options.estimation_model == 'tl':
        result = TLearner(models=_get_regressor(options.estimation_base_model, options))
    elif options.estimation_model == 'cf':
        params = {"max_depth": [5, 10, 20]}
        cf = CausalForestWrapper(n_estimators=1000, random_state=options.seed, n_jobs=1)
        result = GridSearchCV(cf, param_grid=params, n_jobs=options.n_jobs, scoring='neg_mean_squared_error', cv=options.cv)
        fit_type = 'cf'
    else:
        result = _get_regressor(options.estimation_model, options)
        fit_type = 'sklearn'
    return result, fit_type

def _get_ps_weights(x, t, options, eps=0.0001):
    z = np.squeeze(t)
    clf = _get_classifier(options.ipw_model, options)
    clf.fit(x, z)
    e = clf.predict_proba(x).T[1].T + eps
    return z / e + ((1.0 - z) / (1.0 - e))

def estimate(train, test, options):
    X_train = train[0]
    t_train = train[1].flatten()
    y_train = train[2].flatten()
    Xt_train = np.concatenate([X_train, train[1].reshape(-1, 1)], axis=1)
    X_test = test[0]
    
    model, fit_type = _get_model(options)

    if fit_type == 'econml':
        model.fit(Y=y_train, T=t_train, X=X_train)
        te_tr = model.effect(X=X_train, T0=0, T1=1)
        te_test = model.effect(X=X_test, T0=0, T1=1)
        result = te_tr, te_test
    elif fit_type == 'cf':
        model.fit(X=X_train, T=t_train, y=y_train)
        te_tr = model.predict(X_train)
        te_test = model.predict(X_test)
        result = te_tr, te_test
    else: # 'sklearn'
        if 'ipw' in options.estimation_model:
            weights = _get_ps_weights(X_train, t_train, options)
            model.fit(Xt_train, y_train, sample_weight=weights)
        else:
            model.fit(Xt_train, y_train)
        (xta_tr, xtb_tr), (xta_test, xtb_test) = estimation_preproc(train, test)
        ya_tr = model.predict(xta_tr)
        yb_tr = model.predict(xtb_tr)
        ya_test = model.predict(xta_test)
        yb_test = model.predict(xtb_test)
        result = ya_tr, yb_tr, ya_test, yb_test

    fis = None
    if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
        fis = model.best_estimator_.feature_importances_
    elif hasattr(model, 'feature_importances_'):
        fis = model.feature_importances_

    return result, fis

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    random.seed(options.seed)
    np.random.seed(options.seed)

    # Check if output folder exists and create if necessary.
    if not os.path.isdir(options.output_path):
        os.mkdir(options.output_path)

    # Initialise the logger (writes simultaneously to a file and the console).
    init_logger(options)
    logging.debug(options)

    dataset, num_scores = get_dataset(options)

    scores = np.zeros((options.iter, num_scores))
    scores_test = np.zeros((options.iter, num_scores))

    times = np.zeros((options.iter, 1))

    feature_importances = []

    scaler_x = get_scaler(options.scaler)
    scaler_y = get_scaler(options.scaler) if options.scale_y else None

    for i, (train, test, evals) in enumerate(dataset.get_processed_data(merge=True)):
        train, test = dataset.scale_data(train, None, test, (scaler_x, scaler_y), options.scale_bin, options.scale_y)
        
        start_e = time.time()
        estimates, fis = estimate(train, test, options)
        end_e = time.time()
        delta_e = end_e - start_e

        if len(estimates) > 2:
            score, score_test = dataset.evaluate_batch(estimates, scaler_y, evals)
        else:
            score, score_test = dataset.evaluate_batch_effect(estimates, evals)

        scores[i, :], scores_test[i, :] = score, score_test
        dataset.print_scores(i, options.iter, score, score_test)

        if options.save_features and fis is not None:
            feature_importances.append(fis)

        times[i] = delta_e
        logging.info(f"Time elapsed: {delta_e:.3f}s")

    if options.iter > 1:
        logging.info('Total scores')
        logging.info('==============')
        dataset.print_scores_agg(scores, scores_test)
        logging.info(f'Average time: {np.mean(times):.3f}s')
    
    if options.save_results:
        pd.DataFrame(np.concatenate((scores, scores_test), axis=1), columns=dataset.get_score_headers()).to_csv(os.path.join(options.output_path, 'scores.csv'), index=False)
        pd.DataFrame(times, columns=['delta_e']).to_csv(os.path.join(options.output_path, 'times.csv'), index=False)
    
    if options.save_features and feature_importances:
        n_cols = len(feature_importances[0])
        cols = [f'X{i+1}' for i in range(n_cols)]
        # CausalForest doesn't include T in its feature importances.
        if options.estimation_model != 'cf':
            cols[-1] = 'T'
        pd.DataFrame(np.vstack(feature_importances), columns=cols).to_csv(os.path.join(options.output_path, 'feature_importances.csv'), index=False)