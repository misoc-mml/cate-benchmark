import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import sem

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--dtype', type=str, choices=['ihdp', 'jobs', 'news', 'twins'])
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--show', action='store_true', default=False)
    
    parser.add_argument('--sm', type=str, dest='simple_models', nargs='+')
    parser.add_argument('--mm', type=str, dest='meta_models', nargs='+')
    parser.add_argument('--bm', type=str, dest='base_models', nargs='+')

    return parser

def load_results(path, d_name, m_types, columns=None):
    return _process_results(path, d_name, m_types, columns)

def _get_scores(path, d_name, method, columns, score_type):
    try:
        df = pd.read_csv(os.path.join(path, f'{d_name}_{method}', f'{score_type}.csv'))

        if columns is not None:
            df = df[columns]
        
        arr = df.to_numpy()
        mean = np.mean(arr, axis=0)
        err = sem(arr, axis=0) * st.t.ppf(1.95 / 2.0, arr.shape[0] - 1) # 95th CI
        scores = _merge_scores(mean, err)
        result = scores, df
    except:
        result = None

    return result

def _merge_scores(scores1, scores2):
    return [f'{s1:.3f} +/- {s2:.3f}' for s1, s2 in zip(scores1, scores2)]

def _process_results(path, d_name, models, columns):
    m1s, m2s, bms = models
    df_list = []
    for m1 in m1s:
        scores_obj = _get_scores(path, d_name, m1, columns, 'scores')
        times_obj = _get_scores(path, d_name, m1, None, 'times')
        if scores_obj is None:
            continue
        else:
            scores, df = scores_obj
            times, _ = times_obj
        
        df_list.append([m1] + scores + times)

    for m2 in m2s:
        for bm in bms:
            scores_obj = _get_scores(path, d_name, f'{m2}-{bm}', columns, 'scores')
            times_obj = _get_scores(path, d_name, f'{m2}-{bm}', None, 'times')
            if scores_obj is None:
                continue
            else:
                scores, df = scores_obj
                times, _ = times_obj

            df_list.append([f'{m2}-{bm}'] + scores + times)

    cols = ['name'] + list(df.columns) + ['times']

    return pd.DataFrame(df_list, columns=cols)

def get_metrics(d_type):
    if d_type in ('ihdp', 'news'):
        result = ['train ATE', 'train PEHE', 'test ATE', 'test PEHE']
    elif d_type == 'jobs':
        result = ['train ATT', 'train policy', 'test ATT', 'test policy']
    elif d_type == 'twins':
        result = ['train ATE', 'train PEHE', 'train AUC', 'train CF AUC', 'test ATE', 'test PEHE', 'test AUC', 'test CF AUC']
    else:
        raise ValueError('Unrecognised dataset type.')
    
    return result


if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    metrics = get_metrics(options.dtype)
    models = (options.simple_models, options.meta_models, options.base_models)
    df_results = load_results(options.data_path, options.dtype, models, metrics)

    df_results.to_csv(os.path.join(options.output_path, 'combined.csv'), index=False)

    if options.show:
        print(df_results)