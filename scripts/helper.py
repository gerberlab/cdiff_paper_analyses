import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import copy
from sklearn.model_selection import StratifiedKFold
import os
import scipy.stats as st
from collections import defaultdict
import pickle as pkl
from datetime import datetime
from statsmodels.stats.multitest import multipletests

def make_one_hot(x,l=None):
    if l is None:
        l = len(np.unique(x))
    vec = np.zeros(l)
    vec[x] = 1
    return vec

def get_slope_data(dat_dict, weeks, combine_features = 'union', log_transform = False):
    if combine_features is not None:
        if combine_features == 'intersection':
            features = list(set.intersection(*[set(dat_dict[week]['x'].columns.values) for week in weeks]))
        else:
            features = np.unique(np.concatenate([dat_dict[week]['x'].columns.values for week in weeks]))
        x_dat = {week: dat_dict[week]['x'][features].astype(float) for week in weeks}
    else:
        x_dat = {week: dat_dict[week]['x'].astype(float) for week in weeks}
        features = dat_dict[1]['x'].columns.values
    if log_transform:
        if dat_dict[1]['x'].shape[1] == 2509:
            for w,x in x_dat.items():
                x = np.divide(x.T, np.sum(x, 1)).T
                epsilon = get_epsilon(x)
                geom_means = np.exp(np.mean(np.log(x + epsilon), 1))
                transformed = np.divide(x.T, geom_means).T
                epsilon = get_epsilon(transformed)
                x_dat[w] = np.log(transformed + epsilon)
        else:
            epsilon = get_epsilon(x_dat[1])
            x_dat = {key: np.log(value + epsilon) for key, value in x_dat.items()}
    # x_dat = {week: dl.filter_transform(dl.week_raw[key][week]['x'][features],
                                   # targets_by_pt=None, key=key, filter=False) for week in weeks}
    df_times = pd.concat([dat_dict[week]['event_times'] for week in weeks])
    idx = np.unique(df_times.index.values, return_index=True)[1]
    df_times_unique = df_times.iloc[idx]
    pts_drop = df_times_unique.index.values[df_times_unique <= np.max(weeks)]
    df_times = df_times.drop(pts_drop)
    counts = df_times.index.value_counts()
    list_of_dictionaries = [{xx.split('-')[0]: xx.split('-')[1] for xx in dat_dict[week]['x'].index.values} for
                            week in weeks]
    dd = defaultdict(list)

    for d in list_of_dictionaries:
        for k, v in d.items():
            dd[k].append(v)
    slope_data = {}
    y = {}
    etimes = {}
    for pt, count in counts.iteritems():
        if count >= 2:

            slope_data[pt], intercept = np.polyfit(
                [float(i) for i in dd[pt]],
                np.vstack([np.array(x_dat[float(i)].loc[pt + '-' + i].values).astype(float) for i in dd[pt]]), 1).tolist()
            for week in weeks:
                try:
                    y[pt] = dat_dict[week]['y'][pt]
                    etimes[pt] = dat_dict[week]['event_times'][pt]
                    break
                except:
                    continue
    return pd.DataFrame(slope_data, index=features).T, pd.Series(y), pd.Series(etimes)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def bh_corr(pvals, alpha = 0.05):
    pvals = np.array(pvals)
    ix_sort = np.argsort(pvals)
    p_sort = pvals[ix_sort]
    vec = alpha*(np.arange(1, len(pvals)+1)/len(pvals))
    try:
        L = np.max(np.where(p_sort <= vec)[0])
    except:
        L = 0
    corrected = p_sort / (vec/alpha)
    corrected[corrected > 1] = 1
    corrected_f = np.zeros(len(corrected))
    corrected_f[ix_sort] = corrected
    return corrected_f, p_sort[L]

def standardize(x,override = True):
    if not override:
        assert(x.shape[0]<x.shape[1])

    dem = np.std(x,0)
    if (dem == 0).any():
        dem = np.where(np.std(x, 0) == 0, 1, np.std(x, 0))
    stand = (x - np.mean(x, 0))/dem
    ix = np.where((stand == 1).all())[0]
    stand.iloc[:, ix] = 0
    return stand

def get_percentiles(dat, interval = 90, digits = 4, coef = False):
    col = np.array(dat)
    ix_sort = np.argsort(-np.abs(dat))
    sort = col[ix_sort]
    perc = (100-interval)/(2*100)
    ix_upper = np.int(np.floor(len(sort)*perc))
    ix_lower = np.int(np.ceil(len(sort)*(1-perc)))
    if ix_upper >= len(dat):
        ix_upper = ix_upper - 1
    if ix_lower <= len(dat):
        ix_lower = ix_lower - 1
    if coef:
        return (np.round(np.exp(sort[ix_lower]),digits), np.round(np.exp(sort[ix_upper]),digits))
    else:
        return (np.round((sort[ix_lower]),digits), np.round((sort[ix_upper]),digits))


def get_mad(vals, axis = 0):
    return np.median(np.abs(vals - np.mean(vals, axis)), axis)

def get_mad_interval(median, mad, param=None, digits = 4):
    if param == 'coef':
        upper = np.round(np.exp(median + mad),4)
        lower = np.round(np.exp(median - mad),4)
    else:
        upper = np.round(median + mad,4)
        lower  = np.round(median - mad,4)
    return np.round(upper, digits), np.round(lower, digits)

def get_ci(val,z=1.96):
    mean = np.mean(val)
    std = np.std(val)
    l = len(val)
    CI_top = np.round(mean + z*(std)/np.sqrt(l),4)
    CI_bot = np.round(mean - z*(std)/np.sqrt(l),4)
    return (CI_bot, CI_top)

def custom_dist(pt1, pt2, metric):
    if metric == 'e':
        dist = np.linalg.norm(pt1-pt2)
    if metric == 's':
        d1, cc = st.pearsonr(pt1, pt2)
        dist = 1-d1
    return dist

def get_metrics(pred, true, probs):
    ts_pred_in = np.array(pred)
    ts_true_in = np.array(true)
    ts_probs_in = np.array(probs)
    tprr = len(set(np.where(ts_pred_in == 1)[0]).intersection(
        set(np.where(ts_true_in == 1)[0])))/len(np.where(ts_true_in == 1)[0])
    tnrr = len(set(np.where(ts_pred_in== 0)[0]).intersection(
        set(np.where(ts_true_in == 0)[0])))/len(np.where(ts_true_in == 0)[0])
    bac = (tprr + tnrr)/2
    auc_score = sklearn.metrics.roc_auc_score(ts_true_in, ts_probs_in)
    f1 = sklearn.metrics.f1_score(ts_true_in, ts_pred_in)
    ret_dict = {'pred':pred,'true':true,'tpr':tprr,'tnr':tnrr,'bac':bac,'auc':auc_score,'f1':f1,'probs':probs,}
    return ret_dict

def get_class_weights(y, tmpts):
    samp_weights = np.ones(len(y))
    for tmpt in np.unique(tmpts):
        ix_tmpt = np.where(np.array(tmpts)==tmpt)[0]
        # Xt = X_train[ix_tmpt,:]
        yt = y[ix_tmpt]
        ones_ix = np.where(yt == 1)[0]
        zeros_ix = np.where(yt == 0)[0]
        ws = len(yt)/(2* np.bincount([int(x) for x in yt]))

        if len(ws) == 1:
            samp_weights[ix_tmpt] = ws[0]
        elif len(ws) == 2:
            samp_weights[ix_tmpt[ones_ix]] = ws[1]
            samp_weights[ix_tmpt[zeros_ix]] = ws[0]
        else:
            continue
    return samp_weights

def get_log_odds(coefs, names):
    coefs_all = coefs.squeeze()
    ixl0 = np.where(coefs_all < 0)[0]
    ixg0 = np.where(coefs_all > 0)[0]

    g0coefs = copy.deepcopy(coefs_all)
    g0coefs[ixl0] = np.zeros(len(ixl0))

    l0coefs = copy.deepcopy(coefs_all)
    l0coefs[ixg0] = np.zeros(len(ixg0))
    ranks_g = np.argsort(-g0coefs)
    mols_g = names[ranks_g]
    odds_ratio_g = np.exp(coefs_all[ranks_g])

    ranks_l = np.argsort(l0coefs)
    mols_l = names[ranks_l]
    odds_ratio_l = np.exp(coefs_all[ranks_l])
    # import pdb; pdb.set_trace()
    df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
    df = pd.DataFrame(
        df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
    return df

def plot_lambdas_func(lambdict, optim_param, offset, ma, best_lambda):
    fig2, ax2 = plt.subplots()
    ax2.plot(np.array(list(lambdict.keys()))[offset:-offset], ma)
    ax2.axvline(x=best_lambda)
    for ij, k in enumerate(lambdict.keys()):
        ax2.scatter([k],
                    [lambdict[k][optim_param]])
        ax2.set_xlabel('lambda values')

        ax2.set_ylabel(optim_param)
        ax2.set_xscale('log')
        # ax2.set_title('Outer Fold ' + str(ol), fontsize=30)
        if optim_param == 'bac':
            ax2.set_ylim(0, 1)
    return fig2, ax2


def make_toxin_cdiff_targets(labels,dtype,targ_type,step_ahead = 1):
    pt_labels = labels.index.values
    pts = [x.split('-')[0] for x in pt_labels]

    if step_ahead == 0:
        end_dict = {pt_lab: labels[targ_type][pt_lab] for pt_lab in pt_labels}
    else:
        end_dict = {}
        for pt in np.unique(pts):
            ixs = np.where(np.array(pts)==pt)[0][step_ahead:]

            if 'week_one' in dtype or dtype == 'w1_inst':
                labs = [labels.iloc[ix,:][targ_type] for ix in ixs]

                if len(labs)>4:
                    lab = np.round(np.mean(labs[-len(labs):]))
                else:
                    lab = np.round(np.mean(labs[-np.int(np.round(len(labs)/2)):]))
                end_dict[pt+'-1'] = lab

            if 'all_data' in dtype or dtype == 'all_data_inst':
                labs = [labels.iloc[ix,:][targ_type] for ix in ixs]
                labs = pd.DataFrame(labs, index= [labels.index.values[ix-step_ahead] for ix in ixs])
                for i,ix in enumerate(ixs):
                    end_dict[pt_labels[ix-step_ahead]] = np.squeeze(labs.iloc[i])

            # if '16s' in dtype and '.' not in dtype:
            #     import pdb; pdb.set_trace()
            
            else:
                plabs = pt_labels[ixs]
                tmpts = [float(x.split('-')[1]) for x in plabs]
                ix = np.where(np.array(tmpts)==dtype)[0]
                if len(ix)>0:
                    lab = labels.iloc[ix+1,:][targ_type]
                    end_dict[pt_labels[ix][0]]= lab
    
    return pd.DataFrame(end_dict, index = [0])

def dmatrix(data, metric='e'):
    # does linkage among columns
    if metric == 's':
        data_s = data.rank()
        data_s = np.array(data_s)
    else:
        data_s = np.array(data)
    tuples = list(itertools.combinations(range(data_s.shape[0]), 2))
    vec = np.zeros(len(tuples))
    for k, (i, j) in enumerate(tuples):
        if i < j:
            vec[k] = custom_dist(data_s[i, :], data_s[j, :], metric)
    return vec


# def get_vars(data, labels=None, weeks = [0,1,2]):
#     if labels:
#         cleared = data[np.array(labels) == 'Non-recurrer']
#         recur = data[np.array(labels) == 'Recur']
#         within_class_vars = [np.var(cleared, 0), np.var(recur, 0)]
#         class_means = [np.mean(cleared, 0), np.mean(cleared, 0)]
#
#         total_mean = np.mean(data, 0)
#         between_class_vars = 0
#         for i in range(2):
#             between_class_vars += (class_means[i] - total_mean)**2
#
#     else:
#         within_class_vars = None
#         between_class_vars = None
#     if weeks:
#         tmpt = [float(x.split('-')[1]) for x in data.index.values]
#
#     total_vars = np.std(data, 0)/np.abs(np.mean(data,0))
#     vardict = {'within':within_class_vars,'between':between_class_vars,'total':total_vars}
#     return vardict

def filter_vars(data, perc=5, weeks = [0,1,2]):

    if weeks:
        tmpt = [float(x.split('-')[1]) for x in data.index.values]
        rm2 = []
        for week in weeks:
            t_ix = np.where(np.array(tmpt)==week)[0]
            dat_in = data.iloc[t_ix,:]
            variances = np.std(dat_in, 0) / np.abs(np.mean(dat_in, 0))
            rm = np.where(variances > np.percentile(variances, perc))[0].tolist()
            rm2.extend(rm)
        rm2 = np.unique(rm2)
    else:
        variances = np.std(data, 0) / np.abs(np.mean(data, 0))
        rm2 = np.where(variances > np.percentile(variances, perc))[0].tolist()

    temp = data.iloc[:,rm2]
    # import pdb; pdb.set_trace()
    # if len(np.where(np.sum(temp,0)==0)[0]) > 0:
    #     import pdb; pdb.set_trace()
    return data.iloc[:,list(rm2)]

def isclose(a, b, tol=1e-03):
    return (abs(a-b) <= tol).all()

def get_epsilon(data):
    vals = np.array(data).flatten()
    vals[np.where(vals == 0)[0]] = 1
    epsilon = 0.1 * np.min(np.abs(vals))
    return epsilon

def filter_by_train_set(x_train, x_test, meas_key, key = 'metabs', log_transform = True, standardize_data = True):
    if x_train.shape[1]>10:
        filt1 = filter_by_pt(x_train, perc=meas_key['pt_perc'][key], pt_thresh=meas_key['pt_tmpts'][key],
                             meas_thresh=meas_key['meas_thresh'][key], weeks=None)
        filt1_test = x_test[filt1.columns.values]
    else:
        filt1 = x_train
        filt1_test = x_test

    epsilon = get_epsilon(filt1)
    if '16s' in key:
        filt1_test = np.divide(filt1_test.T, np.sum(filt1_test,1)).T
        epsilon = get_epsilon(filt1_test)
        geom_means = np.exp(np.mean(np.log(filt1 + epsilon), 1))
        temp = np.divide(filt1.T, geom_means).T
        epsilon = get_epsilon(temp)
    xout = []
    for x in [filt1, filt1_test]:
        if '16s' not in key:
            if log_transform:
                transformed = np.log(x+ epsilon)
            else:
                transformed = x
        else:
            if log_transform:
                x = np.divide(x.T, np.sum(x,1)).T
                geom_means = np.exp(np.mean(np.log(x + epsilon), 1))
                transformed = np.divide(x.T, geom_means).T
                transformed = np.log(transformed + epsilon)
            else:
                transformed = x
        xout.append(transformed)
    xtr, xtst = xout[0], xout[1]

    if x_train.shape[1]>10:
        filt2 = filter_vars(xtr, perc=meas_key['var_perc'][key], weeks = None)
    else:
        filt2 = xtr
    filt2_test = xtst[filt2.columns.values]
    if standardize_data:
        dem = np.std(filt2,0)
        if (dem == 0).any():
            dem = np.where(np.std(filt2, 0) == 0, 1, np.std(filt2, 0))
        x_train_out = (filt2 - np.mean(filt2, 0))/dem
        x_test_out = (filt2_test - np.mean(filt2,0))/dem
    else:
        x_train_out = filt2
        x_test_out = filt2_test
    return x_train_out, x_test_out

def filter_by_pt(dataset, targets=None, perc = .15, pt_thresh = 1, meas_thresh = 10, weeks = [0,1,2]):
    # tmpts = [float(x.split('-')[1]) for x in dataset.index.values if x.replace('.','').isnumeric()]
    # mets is dataset with ones where data is present, zeros where it is not
    mets = np.zeros(dataset.shape)
    mets[np.abs(dataset) > meas_thresh] = 1


    if weeks is not None:
        df_drop = [x for x in dataset.index.values if not x.split('-')[1].replace('.', '').isnumeric()]
        dataset = dataset.drop(df_drop)
        mets = np.zeros(dataset.shape)
        mets[np.abs(dataset) > meas_thresh] = 1
        pts = [x.split('-')[0] for x in dataset.index.values]
        ixs = dataset.index.values
        ix_add = [i for i in range(len(ixs)) if float(ixs[i].split('-')[1]) in weeks]
        oh = np.zeros(len(pts))
        oh[np.array(ix_add)] = 1
        index = pd.MultiIndex.from_tuples(list(zip(*[pts, oh.tolist()])), names = ['pts', 'add'])
        df = pd.DataFrame(mets, index = index)
        df2 = df.xs(1, level = 'add')
        df2 = df2.groupby(level=0).sum()
        mets = np.zeros(df2.shape)
        mets[np.abs(df2)>0] = 1

    # if measurement of a microbe/metabolite only exists in less than pt_thresh timepoints, set that measurement to zero
    if pt_thresh > 1:
        pts = [x.split('-')[0] for x in dataset.index.values]
        for pt in pts:
            ixs = np.where(np.array(pts) == pt)[0]
            mets_pt = mets[ixs,:]
            # tmpts_pt = np.array(tmpts)[ixs]
            mets_counts = np.sum(mets_pt, 0).astype('int')
            met_rm_ixs = np.where(mets_counts < pt_thresh)[0]
            for ix in ixs:
                mets[ix, met_rm_ixs] = 0

    mets_all_keep = []
    # For each class, count how many measurements exist within that class and keep only measurements in X perc in each class
    if targets is not None:
        if sum(['-' in x for x in targets.index.values]) > 1:
            labels = targets[dataset.index.values]
        else:
            labels = targets[np.array(pts)]
        for lab_cat in np.unique(labels):
            mets_1 = mets[np.where(labels == lab_cat)[0], :]
            met_counts = np.sum(mets_1, 0)
            met_keep_ixs = np.where(met_counts >= np.round(perc * mets_1.shape[0]))[0]
            mets_all_keep.extend(met_keep_ixs)
    else:
        met_counts = np.sum(mets, 0)
        mets_all_keep = np.where(met_counts >= np.round(perc * mets.shape[0]))[0]
    return dataset.iloc[:,np.unique(mets_all_keep)]

def asv_to_name(asv, tax_dat = ['inputs/tax_dat.csv', 'inputs/dada2-taxonomy-silva.csv',
                                'inputs/dada2-taxonomy-rdp.csv']):
    if len(asv)>100:
        classification = []
        for i,td in enumerate(tax_dat):
            if 'tax_dat' in td:
                td_out = pd.read_csv(td, index_col=[0])
                try:
                    met_class = td_out['genus_species'].loc[asv] + ', ' + td_out['OTU'].loc[asv]
                    break
                except:
                    continue
            tdat = pd.read_csv(td)
            td_out = np.array([str(x) for x in tdat[asv]])[-2:]
            td_out = [t for t in td_out if t != 'nan']
            classification.append(' '.join(td_out))

        if len(classification) > 0:
            cl = np.unique(classification)
            if len(cl) > 1:
                cl = cl[0] + ' ; ' + cl[1]
                met_class = cl
            else:
                met_class = cl[0]
    else:
        met_class = asv
    return met_class

def return_taxa_names(sequences, tax_dat = ['inputs/dada2-taxonomy-silva.csv', 'inputs/dada2-taxonomy-rdp.csv']):
    met_class = []
    for metab in sequences:
        if len(metab)>100:
            classification = []
            for i,td in enumerate(tax_dat):
                if 'tax_dat' in td:
                    td_out = pd.read_csv(td, index_col=[0])
                    try:
                        met_class.append(td_out['genus_species'].loc[metab] + ', ' +
                                         td_out['OTU'].loc[metab])
                        break
                    except:
                        continue
                tdat = pd.read_csv(td)
                td_out = np.array([str(x) for x in tdat[metab]])[-2:]
                td_out = [t for t in td_out if t != 'nan']
                classification.append(' '.join(td_out))

            if len(classification)>0:
                cl = np.unique(classification)
                if len(cl) > 1:
                    cl = cl[0] + ' ; ' + cl[1]
                    met_class.append(cl)
                else:
                    met_class.append(cl[0])
        else:
            met_class.append(metab)
    return met_class 


def return_sig_biomarkers(dset_name,data = None, tax_dat = ['inputs/tax_dat.csv'], thresh = 10):
    df_new = pd.read_csv(dset_name) 
  
    # saving xlsx file 
    GFG = pd.ExcelWriter(dset_name.split('.')[0] + '.xlsx') 
    df_new.to_excel(GFG, index = False) 

    GFG.save() 
    xl = pd.ExcelFile(dset_name.split('.')[0] + '.xlsx')
    dset = xl.parse(header = 0, index_col = 0)
    
    lr_l0 = dset.iloc[:,:2]
    lr_l0 = lr_l0[lr_l0['Odds ratio < 1'] < 1]
    
    lr_l0 = lr_l0.iloc[:thresh,:]

    lr_g0 = dset.iloc[:,2:]
    lr_g0 = lr_g0[lr_g0['Odds ratio > 1'] > 1]
    lr_g0 = lr_g0.iloc[:thresh,:]

    metabs = lr_l0
    metabs = np.concatenate((np.array(lr_l0), np.array(lr_g0)))

    if tax_dat is not None:
        met_class = return_taxa_names(metabs[:,0])
                
        if len(metabs[1,0])> 100:
            new_df = pd.DataFrame(np.array([metabs[:,0], metabs[:,1], met_class]).T, columns = ['Biomarker', 'Log Odds', 'Genus Species'] )
        else:
            new_df = pd.DataFrame(metabs, columns = ['Biomarker', 'Log Odds'] )
    else:
        new_df = pd.DataFrame(metabs, columns = ['Biomarker', 'Log Odds'] )
    
    
    if data is not None:
        new_data = data[metabs[:,0]]

        if '16s' in dset_name or 'cdiff' in dset_name:
            new_data = np.divide(new_data.T, np.sum(new_data,1)).T
        return new_df, new_data
    else:
        return new_df


def make_fig(cd, dset_name,label, legend = False, tax_dat = None, thresh = 10):
    
    df_new = pd.read_csv(dset_name) 
  
    # saving xlsx file 
    GFG = pd.ExcelWriter(dset_name.split('.')[0] + '.xlsx') 
    df_new.to_excel(GFG, index = False) 

    GFG.save() 
    xl = pd.ExcelFile(dset_name.split('.')[0] + '.xlsx')
    dset = xl.parse(header = 0, index_col = 0)
    
    lr_l0 = dset.iloc[:,:2]
    lr_l0 = lr_l0[lr_l0['Odds ratio < 1'] < 1]
    
    lr_l0 = lr_l0.iloc[:thresh,:]

    lr_g0 = dset.iloc[:,2:]
    lr_g0 = lr_g0[lr_g0['Odds ratio > 1'] > 1]
    lr_g0 = lr_g0.iloc[:thresh,:]

    metabs = lr_l0
    metabs = np.concatenate((np.array(lr_l0), np.array(lr_g0)))

    
    if tax_dat is not None:
        met_class = return_taxa_names(metabs[:,0], tax_dat = tax_dat)

        metabs[:,0] = np.array(met_class)

        cats = np.unique(np.array(['No Carbon Group']*len(met_class)))
        gp_all = np.array(['No Carbon Group']*len(met_class))
    else:
        gp = []
        for metab in lr_l0['Metabolites']:
            ix = np.where(cd.carbon_gps.isin([metab]))[0]
            if len(ix) > 0:
                gp.append(cd.carbon_gps.index[ix][0])
            else:
                gp.append('No Carbon Group')

        gpg = []
        for metab in lr_g0['Metabolites.1']:
            ix = np.where(cd.carbon_gps.isin([metab]))[0]
            if len(ix) > 0:
                gpg.append(cd.carbon_gps.index[ix][0])
            else:
                gpg.append('No Carbon Group')

        if len(gp) == 0 and len(gpg) ==0:
            return 
        print(label)
        gp_all = gp
        gp_all.extend(gpg)
        cats = np.unique(np.array(gp_all)).tolist()
        try:
            cats.remove('No Carbon Group')
        except:
            return
        cats.append('No Carbon Group')
        cats = np.array(cats)

    fig, ax = plt.subplots(figsize = (30,40))

    matplotlib.rcParams.update({'font.size': 42})

    start = 0
    ticks = []

    metabolites = []
    
    mx = max([np.log(m) for m in metabs[:,1]])
    mn = min([np.log(m) for m in metabs[:,1]])
    
    
    for i, pl in enumerate(cats):
        ix = np.where(np.array(gp_all) == pl)[0]
        try:
            to_plot = np.log(metabs[ix.squeeze(),1].tolist())
        except:
            to_plot = np.log(metabs[ix.squeeze(),1])

        if isinstance(to_plot, float):
            cvec = ['g' if i < 0 else 'b' for i in [to_plot]]
        else:
            cvec = ['g' if i < 0 else 'b' for i in to_plot]

        ax.barh(np.arange(start, start+len(ix)),to_plot, color = cvec)
        ax.barh(np.arange(start, start+len(ix)),to_plot, color = cvec)

        last_start = start
        start = start + len(ix)
        ticks.append((start -2+ last_start )/2 + 0.5)
        if i < len(cats)-1:
            ax.hlines(start -1 + .5,mn,mx, linewidth = 2)
#         else:
#             ax.hlines(start-0.5,-2,np.log(0.6e7), linewidth = 2)

        if len(ix)>1:
            metabolites.extend(metabs[ix.squeeze(),0])
        else:
            metabolites.extend([metabs[ix.squeeze(),0]])


    # import pdb; pdb.set_trace()
    ax.vlines(0,-.5,19.5, linewidth = 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax2 = ax.twinx()
    l = ax.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks2 = f(ax.get_yticks())
    ticks2_means = [np.mean([ticks2[i], ticks2[i+1]]) for i in range(0,len(ticks2)-1)]
    ticks2_fin = zip(ticks2,ticks2_means)
    ticks2_fin2 = np.hstack(np.array([list(x) for x in ticks2_fin]))

    ticks2_fin2 = (ticks2_fin2 - min(ticks2_fin2)+.15)/max(ticks2_fin2 - min(ticks2_fin2)+.15)
    ticks2_fin2 = ticks2_fin2 + -.059

    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks2_fin2))
    ax2.set_yticklabels(metabolites, fontsize = 60)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

#     ax.get_xaxis().set_ticks([])

    if len(cats) > 1:
        ax.tick_params(axis='x', which='major', labelsize=85)
        ax.set_yticks(ticks)
        ax.set_yticklabels(cats, fontsize = 70)

    ax.set_xlabel('Log Odds', fontsize = 75)

    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color='g', lw=10),
                    Line2D([0], [0], color='b', lw=10)]
    
    if legend:
        plt.legend(custom_lines, ['Decreases \n Recurrence Odds', 'Increases \n Recurrence Odds'], 
                loc = 'upper left', fontsize = 70, bbox_to_anchor = (-.2,.83))
        plt.rcParams["legend.loc"] = 'upper right'
        plt.rcParams["legend.bbox_to_anchor"] =  (0.5, 0.5)
        plt.tight_layout()
    plt.savefig(dset_name.split('.')[0] + '.pdf',bbox_inches='tight')
    plt.show()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def merge_16s(filename1, filename2, output_fname):
    xl = pd.ExcelFile(filename1)
    df1 = xl.parse(index_col = 0)
    # new_cols = ['-'.join(x.split('-')[1:]) for x in df1.columns.values]
    # df1.columns = new_cols
    seq1 = df1.index.values

    xl = pd.ExcelFile(filename2)
    df2 = xl.parse(index_col = 0)
    new_cols = ['-'.join(['cdiff', x.split('-')[0], ''.join(x.split('-')[1:])]) for x in df2.columns.values]
    df2.columns = new_cols
    seq2 = df2.index.values

    total_seqs = np.array(np.unique(np.concatenate((seq1,seq2))))

    total_arrs = []
    for seq in total_seqs:
        ix1 = np.where(seq == seq1)[0]
        if len(ix1) > 0:
            arr1 = df1.iloc[ix1,:]
        else:
            arr1 = pd.DataFrame(np.zeros((1, len(df1.iloc[0,:]))), index = [seq], columns = df1.columns.values)
            
        ix2 = np.where(seq == seq2)[0]
        if len(ix2) > 0:
            arr2 = df2.iloc[ix2,:]
        else:
            arr2 = pd.DataFrame(np.zeros((1, len(df2.iloc[0,:]))), index= [seq], columns = df2.columns.values)
        
        total_arrs.append(pd.concat((arr1, arr2), axis = 1))

        total_arrs_f = pd.concat(total_arrs)
        total_arrs_f.to_excel(output_fname, index = True)

def leave_one_out_cv(data, labels, folds = None, ddtype = 'week_one'):
    if ddtype == 'all_data':
        assert(data.shape[0]> 70)
        # import pdb; pdb.set_trace()
        patients = np.array([int(i.split('-')[0]) for i in data.index.values])
        pdict = {}
        for i, pt in enumerate(patients):
            pdict[pt] = labels[i]


        ix_all = []
        for ii in pdict.keys():
            pt_test = ii
            pt_train = list(set(pdict.keys()) - set([ii]))
            ixtrain = (np.concatenate(
                [np.where(patients == j)[0] for j in pt_train]))
            ixtest = np.where(patients == pt_test)[0]
            set1 = set([patients[ix] for ix in ixtest])
            set2 = set([patients[ix] for ix in ixtrain])
            set1.intersection(set2)

            ix_all.append((ixtrain, ixtest))
            assert(not set1.intersection(set2))

    else:
        ix_all = []
        # CHANGE LINE!
        for ixs in range(len(labels)):
            ixtest = [ixs]
            ixtrain = list(set(range(len(labels))) - set(ixtest))
            ix_all.append((ixtrain, ixtest))
    
    if folds is not None:
        if isinstance(labels[0], str):
            cl = np.where(labels == 'Non-recurrer')[0]
            re = np.where(labels == 'Recurrer')[0]
        else:
            cl = np.where(labels == 0)[0]
            re = np.where(labels == 1)[0]
        random_select_cl = np.random.choice(cl, int(folds/2))
        random_select_re = np.random.choice(re, int(folds/2))
        random_select = np.append(random_select_cl, random_select_re)
        ix_all = (np.array(ix_all)[random_select]).tolist()
    return ix_all


# def leave_two_out_combos(data, labels, num_folds=50):
#     if isinstance(labels[0], str):
#         cl = np.where(labels == 'Non-recurrer')[0]
#         re = np.where(labels == 'Recurrer')[0]
#     else:
#         cl = np.where(labels == 0)[0]
#         re = np.where(labels == 1)[0]
#     ixs = np.arange(len(labels))
#     left_out = [list(x) for x in list(itertools.product(cl, re))]
#     ix_keep = [list(set(ixs)-set(l)) for l in left_out]
#     if num_folds is None:
#         ix_all = zip(np.array(ix_keep), np.array(left_out))
#     else:
#         ix_sampled = np.random.choice(np.arange(num_folds), num_folds)
#         ix_all = zip(np.array(ix_keep)[ix_sampled], np.array(left_out)[ix_sampled])
#     return ix_all

def leave_two_out(data, labels, num_folds=50):
    if isinstance(labels[0], str):
        cl = np.where(labels == 'Non-recurrer')[0]
        re = np.where(labels == 'Recurrer')[0]
    else:
        cl = np.where(labels == 0)[0]
        re = np.where(labels == 1)[0]
    ixs = np.arange(len(labels))
    left_out = [list(x) for x in list(itertools.product(cl, re))]
    left_out.extend(list(itertools.combinations(re, 2)))
    ix_keep = [list(set(ixs)-set(l)) for l in left_out]
    if num_folds is None:
        ix_all = zip(np.array(ix_keep), np.array(left_out))
    else:
        ix_sampled = np.random.choice(np.arange(num_folds), num_folds)
        ix_all = zip(np.array(ix_keep)[ix_sampled], np.array(left_out)[ix_sampled])
    return ix_all


def split_to_folds(in_data, in_labels, folds=5, ddtype = 'week_one'):
    # Like split test-train except makes all folds at once

    # If all data, have to take into acct patients w/ multiple timepoints
    in_labels = np.array(in_labels)
    # if ddtype == 'all_data':
    data_perc_take = 1/folds
    patients = np.array([int(ti.split('-')[0])
                        for ti in in_data.index.values])
    unique_patients = np.unique(patients)
    pts_to_take = np.copy(unique_patients)
    ix_all = []
    for f in range(folds):
        # patients to test (1/folds * len(data) patients)
        cleared_gp = patients[np.where(in_labels == 0)[0]]
        recur_gp = patients[np.where(in_labels == 1)[0]]

        cleared_gp = set(cleared_gp) - set(recur_gp)
        recur_gp = set(recur_gp)

        pts_take_cl = np.random.choice(
            list(cleared_gp), int(data_perc_take*len(cleared_gp)))
        pts_take_re = np.random.choice(
            list(recur_gp), int(data_perc_take*len(recur_gp)))

        pts_take = list(set(pts_take_cl) | set(pts_take_re))

        # train with rest
        pts_train = list(set(unique_patients) - set(pts_take))
        ix_ts = np.concatenate(
            [np.where(patients == it)[0] for it in pts_take])

        ix_tr = np.concatenate(
            [np.where(patients == it)[0] for it in pts_train])
        ix_all.append((ix_tr, ix_ts))
    zip_ixs = ix_all
    # If not, can use skf split
    # else:
    #     # patients = in_data.index.values
    #     # for f in range(folds):

    #     skf = StratifiedKFold(folds)
    #     zip_ixs = skf.split(in_data, in_labels)
    return zip_ixs

# def get_resdict_from_file(path):
#     # path = 'out_test/'
#     res_dict = {}
#     for file in os.listdir(path):
#         if file == '.DS_Store':
#             continue
#         key_name = file.split('.')[0].split('_')[0]
#         seed = int(file.split('.')[0].split('_')[1])
#         ix = int(file.split('.')[0].split('_')[2])
#         with open(inner_path + file, 'rb') as f:
#             # temp is a dictionary with parameter keys
#             temp = pkl.load(f)
#             if key_name not in res_dict.keys():
#                 res_dict[key_name] = {}
#             if ix not in res_dict[key_name].keys():
#                 res_dict[key_name][ix] = {}
#             for test_param in temp[ix].keys():
#                 if test_param not in res_dict[key_name][ix].keys():
#                     res_dict[key_name][ix][test_param] = {}
#                 # res dict keys go model, held out test index, parameter to test, seed
#                 res_dict[key_name][ix][test_param][seed] = temp[ix][test_param]
#     return res_dict

def get_best_param(res_dict, key_name):
    best_param = {}
    for ix in res_dict[key_name].keys():
        all_aucs = {}
        auc = {}
        params = list(res_dict[key_name][ix].keys())
        for param in params:
            if param not in auc.keys():
                auc[param] = []
            for seed in res_dict[key_name][ix][param].keys():
                auc[param].append(res_dict[key_name][ix][param][seed]['metrics']['auc'])
        
            all_aucs[param] = np.median(auc[param])
                
        vec = np.array([all_aucs[p] for p in params])
        ma = moving_average(vec, n=5)
        offset = int(np.floor(5./2))
        max_ix = np.argmax(ma) + offset
        
        best_param[ix] = params[max_ix]
    return best_param


def get_coefs(in_path, folder):
    coef_res_dict = {}
    coef_res_dict[folder] = {}
    print(in_path + folder)
    for file in os.listdir(in_path + folder):
        if 'coef' not in file or 'pkl' not in file:
            continue
        if file == '.DS_Store':
            continue
        seed = int(file.split('_')[-1].split('.')[0])
        with open(in_path + folder + '/' + file, 'rb') as f:
            # temp is a dictionary with parameter keys
            temp = pkl.load(f)
        coef_res_dict[folder][seed] = temp
    #             get_coefs(coef_res_dict, path, model)
    model = folder
    if len(coef_res_dict[model].keys()) == 0:
        return
    if 'data' in coef_res_dict[model][0].keys():
        if isinstance(coef_res_dict[model][0]['data'][0], list):
            coef_names = np.concatenate([x.columns.values for x in coef_res_dict[model][0]['data'][0]])
        else:
            try:
                coef_names = coef_res_dict[model][0]['data'].columns.values[:-2]
            except:
                coef_names = coef_res_dict[model][0]['data'][0].columns.values
    else:
        coef_names = coef_res_dict[model][0]['x'].columns.values
    if 'outcome' in coef_names:
        coef_names = coef_names[:-2]
    seeds = list(coef_res_dict[model].keys())
    if 'importances' in coef_res_dict[model][0].keys():
        seeds = list(coef_res_dict[model].keys())
        coef_arr = np.array([coef_res_dict[model][i]['importances'] for i in seeds]).squeeze()
    else:
        if 'model' in coef_res_dict[model][0].keys():
            m = 'model'
        elif 'best_model' in coef_res_dict[model][0].keys():
            m = 'best_model'

        try:
            seeds = list(coef_res_dict[model][0][m].keys())
        except:
            seeds = np.arange(len(coef_res_dict[model][0][m]))

        try:
            try:
                ix = np.argmax(list(zip(*[coef_res_dict[folder][0][m][i].coef_.shape for i in seeds]))[1])
                coefs = coef_res_dict[model][0][m][ix].coef_
            except:
                coefs = coef_res_dict[model][0][m][0].feature_importances_
        except:
            ix = np.argmax(list(zip(*[coef_res_dict[folder][0][m][i].shape for i in seeds]))[1])
            coefs = coef_res_dict[model][0][m][ix]
        if len(coefs.squeeze().shape) > 1:
            coef_arr = np.array([coef_res_dict[model][0][m][i].coef_[:, -1] for i in seeds]).squeeze()
        else:
            try:
                try:
                    coef_arr = np.array([coef_res_dict[model][0][m][i].coef_ for i in seeds]).squeeze()
                except:
                    coef_arr = np.array([coef_res_dict[model][0][m][i].feature_importances_ for i in seeds]).squeeze()
            except:
                coef_arr = np.array([coef_res_dict[model][0][m][i] for i in seeds]).squeeze()

    if 'cox' in in_path and coef_arr.shape[1] > len(coef_names):
        coef_arr = coef_arr[:, :-2]
    median = np.median(coef_arr, 0)
    mad = get_mad(coef_arr, 0)
    upper, lower = get_mad_interval(median, mad, 'coef')
    lower_perc, upper_perc = list(zip(*[(np.round(np.exp(get_percentiles(coef_arr[:, i], 95)), 8))
                                        for i in np.arange(coef_arr.shape[-1])]))

    lower_perc90, upper_perc90 = list(zip(*[(np.round(np.exp(get_percentiles(coef_arr[:, i], 90)), 8))
                                            for i in np.arange(coef_arr.shape[-1])]))

    lower_perc75, upper_perc75 = list(zip(*[(np.round(np.exp(get_percentiles(coef_arr[:, i], 75)), 8))
                                            for i in np.arange(coef_arr.shape[-1])]))

    median_lo = np.round(np.exp(median), 8)
    mad_lo = np.round(np.exp(mad), 8)
    med_cdi_lo = list(zip(lower, upper))
    fin_dict = {'Median OR': median_lo,
                # 'MAD interval': med_cdi_lo,
                '95% Interval': list(zip(lower_perc, upper_perc)),
                '90% Interval': list(zip(lower_perc90, upper_perc90)),
                '75% Interval': list(zip(lower_perc75, upper_perc75))}
    coef_df = pd.DataFrame(fin_dict, index=coef_names)

    ix_median = np.argsort(-np.abs(median))
    #     ix_median_nzero = [ix for ix in ix_median if median[ix]!=0]

    #     ix_MAD = np.argsort(-np.abs(mad))
    #     ix_MAD_all = [ix for ix in ix_MAD if ix not in ix_median_nzero]
    #     ix_tot = np.concatenate([ix_median_nzero, ix_MAD_all])
    ix_tot = ix_median

    coef_df = coef_df.iloc[ix_tot]

    coef_df.to_csv(in_path + folder + '/' + 'SigBiomarkers_' + model + str(datetime.now()).split(' ')[0] + '.csv')

def get_auc_df(final_res_dict, out_path, use_seeds = False):
    auc_dict = {}
    auc_dict_2 = {}
# out_path_auc = '/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/ML_results/' + model_run + '/'
    for model in final_res_dict.keys():
#         out_path_auc = path + model + '/'
        if 'score' in final_res_dict[model].keys():
            aucs = final_res_dict[model]['score']
        else:
            med_aucs = {}
            ixs = list(final_res_dict[model].keys())
            try:
                ix0 = ixs[0]
            except:
                continue
            if use_seeds:
                seeds = list(final_res_dict[model][ixs[0]].keys())
                for ix in ixs:
                    seeds = list(final_res_dict[model][ix0].keys())
                    if 'metrics' in final_res_dict[model][ix0][seeds[0]].keys():
                        med_aucs[ix] = np.median([final_res_dict[model][ix][s]['metrics']['auc'] for s in seeds])
                    elif 'ci' in final_res_dict[model][ix0][seeds[0]].keys():
                        med_aucs[ix] = np.median([final_res_dict[model][ix][s]['ci'] for s in seeds])
                    elif 'auc' in final_res_dict[model][ix0][seeds[0]].keys():
                        med_aucs[ix] = np.median([final_res_dict[model][ix][s]['aucs'] for s in seeds])
                    else:
                        print('no aucs')
                        break
                aucs = [med_aucs[ix] for ix in ixs]
            else:
                if 'metrics' in final_res_dict[model][ix0].keys():
                    aucs = [final_res_dict[model][ix]['metrics']['auc'] for ix in ixs]
                elif 'ci' in final_res_dict[model][ix0].keys():
                    aucs = [final_res_dict[model][ix]['ci'] for ix in ixs]
                elif 'auc' in final_res_dict[model][ix0].keys():
                    aucs = [final_res_dict[model][ix]['auc'] for ix in ixs]
                elif 'score' in final_res_dict[model][ix0].keys():
                    aucs = [final_res_dict[model][ix]['score'] for ix in ixs]
                else:
                    print('no aucs')
                    break
                if isinstance(aucs[0], list):
                    auc_arr = np.array(aucs)
                    aucs = [np.median(auc_arr[:,i]) for i in np.arange(auc_arr.shape[1])]
                    importances = [final_res_dict[model][ix]['importances'] for ix in ixs]
        num_folds = len(aucs)
        median = np.median(aucs)
        mad = get_mad(aucs)
        upper, lower = get_mad_interval(median, mad, 'auc')
        CI = get_percentiles(aucs, interval = 95, coef = False)

        mname = model + ', ' + str(num_folds) +' folds'
        auc_dict[mname] = {'Median':median, 'MAD Interval': str((lower, upper)),\
                           '95% Interval': str(CI)}
        auc_dict_2[model] = {'Median':median, 'MAD Interval': str((lower, upper)),\
                           '95% Interval': str(CI), 'data': aucs}

    auc_df = pd.DataFrame(auc_dict, index = ['Median', 'MAD Interval', '95% Interval']).T
    auc_df.to_csv(out_path + 'auc_'+ str(datetime.now()).split(' ')[0] + '.csv')
    return auc_dict_2

def get_panal_locs(ax):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    return (xmin,ymax + 0.1*(ymax-ymin))


import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, Rectangle


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors, shapes):
        self.colors = colors
        self.shapes = shapes


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            if orig_handle.shapes[i] == 's':
                patches.append(Rectangle([width / len(orig_handle.colors) * i, 0],
                                         width / len(orig_handle.colors),
                                         height,
                                         facecolor=c,
                                         edgecolor='gray'))
            else:
                patches.append(
                    Ellipse([(width / len(orig_handle.colors) * i) + width / len(orig_handle.colors) / 2, height / 2],
                            width / len(orig_handle.colors),
                            height,
                            facecolor=c,
                            edgecolor='gray'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch

def get_masked_arr(d, ix):
    da = d[ix,:]
    db = da[:,ix]
    return db

# plot metabolites
def plot_metab_over_time(metab, pval, dat, path):
    df, y, y_per_pt = dat['filtered_data'], dat['targets'], dat['targets_by_pt']

    ix_keep = [ix for ix in df.index.values if ix.split('-')[1].split('.')[0].isnumeric()]
    df, y = df.loc[ix_keep, :], y.loc[ix_keep]

    if metab not in df.columns.values:
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    pts = [yy.split('-')[0] for yy in y.index.values]
    tmpts = [yy.split('-')[1] for yy in y.index.values]
    for pt in np.unique(pts):
        ix = np.where(np.array(pts) == pt)[0]
        pt_tmpts = np.array([float(t) for t in np.array(tmpts)[ix]])

        if y_per_pt[pt] == 'Recurrer':
            ax.plot(pt_tmpts + .08, df[metab].iloc[ix], '*', c='r', ms=10, alpha=0.5)
        else:
            ax.plot(pt_tmpts - .08, df[metab].iloc[ix], 'o', c='g', mfc='none', alpha=0.5)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin - 10, ymax])

    m_zero = np.min(df[metab])
    m_val = np.max(df[metab])
    for tmpt in np.unique(tmpts):
        if float(tmpt) > 4:
            continue
        ix = np.where(np.array(tmpts) == tmpt)[0]

        ix_both = list(set(ix).intersection(set(np.where(y == "Recurrer")[0])))
        box1 = ax.boxplot(df[metab].iloc[ix_both], positions=[float(tmpt) + .08])
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box1[element], color='r', linewidth=5, alpha=0.3)

        ix_both = list(set(ix).intersection(set(np.where(y == "Non-recurrer")[0])))
        box2 = ax.boxplot(df[metab].iloc[ix_both], positions=[float(tmpt) - .08])
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box2[element], color='g', linewidth=5, alpha=0.3)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticks([0, 2.5], minor=True)
    ax.grid(False, axis='x')
    ax.grid(False, axis='x', which='minor')

    ax.set_xticklabels(labels=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                       # rotation = 10,  ha = 'center',
                       minor=False, fontsize=12)
    ax.set_xticklabels(labels=['\nPre-Antibiotic Treatment', '\nPost-Antibiotic Treatment'],
                       minor=True, fontsize=16)

    ax.set_ylabel('Standardized Value', fontsize=18)

    if len(metab) > 100:
        metab = asv_to_name(metab)
    metab = metab.replace('/', '_')
    plt.title(metab + ', p=' + str(pval))
    plt.xlim([-.25, 4.25])
    plt.ylim([m_zero - .1, m_val + 1.1])
    plt.savefig(path + metab + '.pdf')
    plt.close()


def get_data(key, dl, dtype='filtered_data', week=None, features=None):
    data = dl.keys[key][dtype]
    ix_keep = [ix for ix in data.index.values if ix.split('-')[1].isnumeric()]
    if week == None:
        ix_keep = [ix for ix in ix_keep if float(ix.split('-')[1]) <= 2]
    else:
        ix_keep = [ix for ix in ix_keep if float(ix.split('-')[1]) == week]

    data = data.loc[ix_keep]
    if features is not None:
        data = data[features]
    outcomes = dl.keys[key]['targets'].loc[ix_keep]
    return data, outcomes

def to_distance_matrix(tree):
    allclades = list(tree.find_clades(order="level"))
    lookup = {}
    for i, elem in enumerate(allclades):
        lookup[elem] = i
    distmat = {}
    clade_names = []
    for parent in tree.find_clades():
        if parent.name:
            for child in tree.find_clades():
                if child.name:
                    if child.name.split(', ')[-1] not in distmat.keys():
                        distmat[child.name.split(', ')[-1]] = {}
                    if parent.name.split(', ')[-1] not in distmat.keys():
                        distmat[parent.name.split(', ')[-1]] = {}

                    distmat[parent.name.split(', ')[-1]][child.name.split(', ')[-1]] = tree.distance(parent, child)
                    distmat[child.name.split(', ')[-1]][parent.name.split(', ')[-1]] = tree.distance(parent, child)

    return pd.DataFrame(distmat)