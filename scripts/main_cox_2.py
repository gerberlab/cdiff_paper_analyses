from helper import *
from datetime import datetime
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import os
import time
import itertools
from dataLoader import *
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
warnings.filterwarnings("ignore")


def train_cox(x, y, event_times, xadd = None, outer_split = leave_two_out, inner_split = leave_two_out, num_folds = None,
              meas_key = None, key = 'metabs'):
    if num_folds is None:
        print('none')
    else:
        print(num_folds)
    np.random.seed(5)
    # if feature_grid is None:
    #     feature_grid = np.logspace(7, 20, 14)
    hazards = []
    event_times = []
    event_outcomes = []
    score_vec = []
    model_out_dict = {}
    ix_inner = outer_split(x, y, num_folds=100)
    lambda_dict = {}
    for ic_in, ix_in in enumerate(ix_inner):
        train_index, test_index = ix_in
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        e_train, e_test = event_times[train_index], event_times[test_index]
        kk = key.split('_')[0]
        if (x_train<0).any().any() or kk == 'demo':
            x_train, x_test = filter_by_train_set(x_train,
                                                    x_test,
                                                    meas_key, key = key, log_transform = False)
        else:
            x_train, x_test = filter_by_train_set(x_train,
                                                    x_test,
                                                    meas_key, key=key, log_transform=True)

        if np.sum(y_test.values)<1:
            continue

        if xadd is not None:
            x_tr_ls = [x_train]
            x_ts_ls = [x_test]
            for ii,xx in enumerate(xadd):
                kk = key.split('_')[ii+1]
                xx_train, xx_test = xx.iloc[train_index, :], xx.iloc[test_index, :]
                if (xx_train < 0).any().any() or kk == 'demo':
                    xx_train, xx_test = filter_by_train_set(xx_train, xx_test, meas_key, key=kk, log_transform=False)
                else:
                    xx_train, xx_test = filter_by_train_set(xx_train, xx_test, meas_key, key=kk, log_transform=True)
                x_tr_ls.append(xx_train)
                x_ts_ls.append(xx_test)
            feats = np.concatenate([xx.columns.values for xx in x_tr_ls])
            x_train = pd.DataFrame(np.concatenate(x_tr_ls, axis = 1), columns = feats, index = x_tr_ls[0].index.values)
            x_test = pd.DataFrame(np.concatenate(x_ts_ls, axis = 1), columns = feats, index = x_ts_ls[0].index.values)


        # x_train_ = x_train.drop(['week','outcome'], axis = 1)
        yy = list(zip(y_train, e_train))
        y_arr = np.array(yy, dtype = [('e.tdm', '?'), ('t.tdm', '<f8')])

        ix_inner2 = inner_split(x_train, y_train, num_folds = 100)
        lamb_dict = {}
        lamb_dict['auc'] = {}
        lamb_dict['ci'] = {}
        model2 = CoxnetSurvivalAnalysis(l1_ratio=1)

        model_dict = {}
        alphas = None
        hazards_dict = {}
        e_times_dict = {}
        e_outcomes_dict = {}
        score_dict = {}

        coxnet_pipe = CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=0.001,n_alphas=300)

        coxnet_pipe.fit(x_train, y_arr)
        alphas = coxnet_pipe.alphas_

        for ic_in2, ix_in2 in enumerate(ix_inner2):
            start_inner = time.time()

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]
            y_tr2, y_ts2 = y_train.iloc[train_ix], y_train.iloc[test_ix]
            e_tr2, e_ts2 = e_train.iloc[train_ix], e_train.iloc[test_ix]

            if np.sum(y_tr2) < 1:
                continue

            yy_test = list(zip(y_ts2, e_ts2))
            yy_test_arr = np.array(yy_test, dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
            if len(np.unique(yy_test_arr)) < len(test_ix):
                continue

            # week = x_tr2['week']
            # outcome = x_tr2['outcome']
            if (y_tr2 == 0).all():
                continue
            # x_tr2_ = x_tr2.drop(['week', 'outcome'], axis=1)
            yy2 = list(zip(y_tr2, e_tr2))
            y_arr2 = np.array(yy2, dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
            model2.set_params(alphas=alphas)
            try:
                model2.fit(x_tr2, y_arr2)
            except:
                print('removed alpha ' + str(alphas[0]))
                alphas_n = np.delete(alphas, 0)
                model2.set_params(alphas=alphas_n)
                while(1):
                    try:
                        model2.fit(x_tr2, y_arr2)
                        alphas = alphas_n
                        break
                    except:
                        print('removed alpha ' + str(alphas_n[0]))
                        alphas_n = np.delete(alphas, 0)
                        model2.set_params(alphas=alphas_n)
                    if len(alphas_n)<=2:
                        break
                if len(alphas_n)<=2:
                    continue
            # alphas_new = model2.alphas_
            # if ic_in2 == 0:
            #     alphas = alphas_new

            model_dict[ic_in2] = model2
            for i, alpha in enumerate(alphas):
                if i not in hazards_dict.keys():
                    hazards_dict[i]={}
                    e_times_dict[i] = {}
                    e_outcomes_dict[i] = {}
                    score_dict[i] = {}
                risk_scores = model2.predict(x_ts2, alpha=alpha)
                hazards_dict[i][ic_in2] = risk_scores
                e_times_dict[i][ic_in2] = e_ts2
                e_outcomes_dict[i][ic_in2] = y_ts2

                if len(test_ix)>=2:
                    try:
                        ci = concordance_index_censored(e_outcomes_dict[i][ic_in2].astype(bool), e_times_dict[i][ic_in2],
                                                                           hazards_dict[i][ic_in2])[0]
                    except:
                        print('debug')
                        print(y_ts2)
                        print(e_ts2)
                        print('')
                        continue

                    if not np.isnan(ci):
                        score_dict[i][ic_in2] = ci


        if len(score_dict[i]) > 0:
            scores = {i: sum(score_dict[i].values())/len(score_dict[i].values()) for i in score_dict.keys()}
        else:
            scores = {}
            for a_ix in hazards_dict.keys():
                alpha_num = alphas[a_ix]
                scores[alpha_num], concordant, discondordant, tied_risk, tied_time = concordance_index_censored(
                    np.array(np.concatenate(list(e_outcomes_dict[a_ix].values()))).astype(bool),
                    np.array(np.concatenate(list(e_times_dict[a_ix].values()))),
                    np.array(np.concatenate(list(hazards_dict[a_ix].values()))))

        lambdas, aucs_in = list(zip(*scores.items()))
        ix_max = np.argmax(aucs_in)
        best_lamb = alphas[ix_max]

        lambda_dict[ic_in] = {'best_lambda': best_lamb, 'scores': scores, 'event_outcomes':event_outcomes, 'times':event_times,
                       'hazards':hazards, 'lambdas_tested': alphas}
        model_out = CoxnetSurvivalAnalysis(l1_ratio = 1, alphas = alphas)

        model_out.fit(x_train, y_arr)

        risk_scores = model_out.predict(x_test, alpha = best_lamb)

        hazards.append(risk_scores)
        event_times.append(e_test)
        event_outcomes.append(y_test)

        coefs = model_out.coef_[:,ix_max]
        if xadd is not None:
            xix = x.columns.values.tolist()
            xix.extend(np.concatenate([xx.columns.values.tolist() for xx in xadd]))
            out_df = pd.DataFrame({'odds_ratio':np.zeros(x.shape[1] + np.sum([xx.shape[1] for xx in xadd]))},
                                  index = xix)
        else:
            out_df = pd.DataFrame({'odds_ratio': np.zeros(x.shape[1])},
                              index=x_train.columns.values)
        out_df.loc[x_train.columns.values] = np.expand_dims(coefs,1)

        model_out_dict[ic_in] = out_df
        if len(test_index) > 1:
            ci = concordance_index_censored(y_test.astype(bool), e_test, risk_scores)[0]
            if not np.isnan(ci):
                score_vec.append(ci)

    if len(score_vec) > 1:
        score = sum(score_vec)/len(score_vec)
    else:
        score, concordant, discondordant, tied_risk, tied_time = concordance_index_censored(
            np.array(np.concatenate(event_outcomes)).astype(bool),
            np.array(np.concatenate(event_times)), np.array(np.concatenate(hazards)))

    final_dict = {}
    final_dict['score'] = score
    final_dict['model'] = model_out_dict
    final_dict['hazards'] = hazards
    final_dict['event_times'] = event_times
    final_dict['event_outcomes'] = event_outcomes
    final_dict['lambdas'] = lambda_dict
    return final_dict


if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-ix", "--ix", help="index for splits", type=int)
    parser.add_argument("-o", "--o", help="outpath", type=str)
    parser.add_argument("-i", "--i", help="inpath", type=str)
    parser.add_argument("-type", "--type", help="inpath", type=str)
    parser.add_argument("-week", "--week", help="week", type=str)
    parser.add_argument("-seed", "--seed", help="seed", type=int)
    parser.add_argument("-folds", "--folds", help="use_folds", type=int)
    parser.add_argument("-num_folds", "--num_folds", help="num_folds", type=int)
    args = parser.parse_args()

    if args.ix is None:
        args.ix = 0
    if args.o is None:
        args.o = 'test_lr_fast'
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
    if args.i is None:
        args.i = 'metabs_16s_scfa_toxin_demo'
    if args.type is None:
        args.type = 'auc'
    if args.week is None:
        args.week = 1
    else:
        args.week = [float(w) for w in args.week.split('_')]
    if args.folds is None:
        args.folds = 0
    if args.num_folds is None:
        args.num_folds = None

    np.random.seed(args.seed)
    if isinstance(args.week, list):
        if len(args.week)==1:
            args.week = args.week[0]

    dl = dataLoader(pt_perc=0, meas_thresh=0, var_perc=0, pt_tmpts = 1)
    meas_key = {'pt_perc':{'metabs':0.25, '16s':0.1, 'scfa': 0, 'toxin': 0},
                'meas_thresh':{'metabs':0, '16s':10,'scfa':0,'toxin':0},
                'var_perc':{'metabs':50, '16s':5,'scfa':0,'toxin':0},
                'pt_tmpts':{'metabs':1, '16s':1,'scfa':1,'toxin':1}}
    if isinstance(args.week, list):
        dat_dict = dl.week_raw[args.i]
        x, y, event_times = get_slope_data(dat_dict, args.week, log_transform = True)
    else:
        xls = None
        if '_' in args.i:
            ii = args.i.split('_')
            xls = []
            for ai in ii:
                if ai == 'demo':
                    data = dl.week_raw['metabs'][args.week]
                    x = dl.clinical
                    y, event_times = data['y'], data['event_times']
                else:
                    data = dl.week_raw[ai][args.week]
                    x, y, event_times = data['x'], data['y'], data['event_times']
                if isinstance(x.index.values[0], str):
                    x.index = [xind.split('-')[0] for xind in x.index.values]
                xls.append(x)
            xix = list(set.intersection(*[set(xx.index.values) for xx in xls]))
            xls_t = [x.loc[xix] for x in xls]
            y = y[xix]
            event_times = event_times[xix]
            x = xls_t[0]
            xls = xls_t[1:]
        else:
            if args.i == 'demo':
                x = dl.demographics
                y = dl.week['metabs'][1]['y']
                event_times = dl.week['metabs'][1]['event_times']
            else:
                data = dl.week_raw[args.i][args.week]
                x, y, event_times = data['x'], data['y'], data['event_times']
            x.index = [xind.split('-')[0] for xind in x.index.values]
    y = (np.array(y) == 'Recurrer').astype('float')

    path_out = args.o + '/' + args.i + '/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    np.random.seed(args.seed)
    if args.type == 'auc':
        ixs = leave_one_out_cv(x, y)
        train_index, test_index = ixs[args.ix]
        x_train0, x_test0 = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train0, y_test0 = y[train_index], y[test_index]
        e_train, e_test = event_times[train_index], event_times[test_index]
        xls_train = None
        if xls is not None:
            xls_train = []
            for xx in xls:
                xx_train0, xx_test0 = xx.iloc[train_index, :], xx.iloc[test_index, :]
                xls_train.append(xx_train0)

        final_res_dict = train_cox(x_train0, y_train0, e_train, xadd = xls_train, num_folds=args.num_folds, meas_key = meas_key, key = args.i)
            # final_res_dict = train_cox(x_train0)
    elif args.type == 'coef':
        final_res_dict = train_cox(x, y, event_times, xadd = xls, num_folds=args.num_folds, meas_key = meas_key, key = args.i)
            # final_res_dict = train_cox(x)

    if xls is not None:
        final_res_dict['data'] = (x, y, event_times)
    else:
        final_res_dict['data'] = (xls_t, y, event_times)

    with open(path_out + args.type + '_ix_' + str(args.ix)+ '.pkl', 'wb') as f:
        pkl.dump(final_res_dict, f)

    end = time.time()
    passed = np.round((end - start) / 60, 3)
    f2 = open(args.o + '/' + args.i + ".txt", "a")
    try:
        f2.write('index ' + str(args.ix) + ', CI ' + str(final_res_dict['score']) +' in ' + str(passed) + ' minutes' + '\n')
    except:
        f2.write(
            'index ' + str(args.ix) + ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()

# Get coefficients from nested CV
