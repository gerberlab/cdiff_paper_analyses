import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.decomposition import PCA
import matplotlib
import sklearn
import copy
from seaborn import clustermap
from scipy.cluster.hierarchy import linkage
import random
from random import randint
from sklearn.manifold import MDS
# beta_diversity = 1
# # import skbio
# from skbio.diversity import beta_diversity
import seaborn as sns
from matplotlib.collections import LineCollection


import scipy
from helper import *
from matplotlib import cm
from dataLoader import *
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
# from statsmodels.graphics.gofplots import qqplot

# univariate analysis for metabolites and scfas
def rank_sum(x, targets, method = 'ranksum', cutoff=.05):
    if isinstance(targets[0], str):
        targets = (np.array(targets) == 'Recurrer').astype('float')
    else:
        targets = np.array(targets)
    pval = []
    teststat = []
    for i in range(x.shape[1]):
        xin = np.array(x)[:, i]
        X = xin[targets == 1]
        Y = xin[targets == 0]
        # xin1 = (xin - np.min(xin,0))/(np.max(xin,0)-np.min(xin,0))
        if method == 'ranksum':
            s, p = st.ranksums(X, Y)
        elif method == 'kruskal':
            try:
                s, p = st.kruskal(X,Y)
            except:
                p = 1
        elif method == 'ttest':
            s, p = st.ttest_ind(X,Y)
        pval.append(p)
        teststat.append(s)

    pval = np.array(pval)
    pval[np.isnan(pval)] = 1
    # corrected, alpha = bh_corr(np.array(pval), .05)

    reject, corrected, a1, a2 = multipletests(pval, alpha=.05, method='fdr_bh')
    df = pd.DataFrame(np.vstack((pval, corrected, teststat)).T, columns=[
        'P_Val', 'BH corrected', 't-stat'], index=x.columns.values)

    return df.sort_values('P_Val', ascending=True)

# Entropy significance analysis between timepoings, within each outcome group
def entropy_over_time(entropy_df, tmpts = [0,1,2], use_ranksum = False, use_one_sided = True,
                      path_to_save = '/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/'):
    pval = {}
    pvals = []
    for cl in ['Recurrer','Non-recurrer']:
        pval[cl] = {}
        for tix, tmpt in enumerate(tmpts[:-1]):
            entropy_1 = entropy_df[tmpt]
            entropy_2 = entropy_df[tmpt + 1]
            entropy_1 = entropy_1.loc[entropy_1['Outcome'] == cl]
            entropy_2 = entropy_2.loc[entropy_2['Outcome'] == cl]
            ix_sim = set(entropy_1.index.values).intersection(set(entropy_2.index.values))
            if use_one_sided:
                if tmpt == 0:
                    alt = 'greater'
                else:
                    alt = 'less'
            else:
                alt = 'two-sided'
            try:
                if use_ranksum:
                    _, p = st.wilcoxon(entropy_1.loc[ix_sim]['Week ' + str(tmpt)],
                                                              entropy_2.loc[ix_sim]['Week ' + str(tmpt+1)],
                                                             alternative = alt)
                    tname = 'wilcoxnon_ranksum'
                else:
                    _, p = st.mannwhitneyu(entropy_1['Week ' + str(tmpt)],  entropy_2['Week ' + str(tmpt+1)],
                                                             alternative = alt)
                    tname = 'mannwhitney'

                pval[cl][(tmpt, tmpt+1)] = p
                pvals.append(p)
            except:
                continue
    pval_corr = {}
    pvals_corr = multipletests(pvals, alpha = 0.05, method = 'fdr_bh')[1]
    it = 0
    for cl in ['Recurrer','Non-recurrer']:
        pval_corr[cl] = {}
        for tmpt in tmpts[:-1]:
            pval_corr[cl][(tmpt, tmpt+1)] = pvals_corr[it]
            it += 1
    # p_df = pd.DataFrame(pval, index = [0]).T
    # p_df.to_csv('paper_figs/entropy_ttest.csv')
    dictionary = {'Uncorrected': pval, 'Corrected': pval_corr}
    reform = {(outerKey, innerKey): values for outerKey,
              innerDict in dictionary.items() for innerKey, values in innerDict.items()}
    df_over_time = pd.DataFrame(reform)
    pd.DataFrame(reform).to_csv(path_to_save + 'Fig3_results/entropy/intra_entropy_' + tname + '_' + alt + '.csv')
    return df_over_time

# entropy significance analysis between outcome groups
def entropy_bw_grps(entropy_df, tmpts = [0,1,2], use_one_sided = True,
                   path_to_save = '/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/'):
    pval = {}
    outcome= ['Recurrer','Non-recurrer']
    for tmpt in tmpts:
        entropy = entropy_df[tmpt]
        entropy_1 = entropy.loc[entropy['Outcome'] == outcome[0]]
        entropy_2 = entropy.loc[entropy['Outcome'] == outcome[1]]

        if use_one_sided:
            if tmpt == 0:
                alt = 'two-sided'
            else:
                alt = 'less'
        else:
            alt = 'two-sided'

        try:
            _, pval[tmpt] = st.mannwhitneyu(entropy_1['Week ' + str(tmpt)],
                                                      entropy_2['Week ' + str(tmpt)],
                                                     alternative = alt)
        except:
            continue
    pval_corr = {t: p for t, p in zip(tmpts, multipletests(list(pval.values()), alpha = 0.05, method = 'fdr_bh')[1])}

    # p_df = pd.DataFrame(pval, index = [0]).T
    # p_df.to_csv('paper_figs/entropy_ttest.csv')
    df_cl_v_re = pd.DataFrame({'Uncorrected': pval, 'Corrected': pval_corr})
    pd.DataFrame({'Uncorrected': pval, 'Corrected': pval_corr}).to_csv(path_to_save +
                                                                       'Fig3_results/entropy/inter_entropy_mann_whit_' +
                                                                       alt + '.csv')
    return df_cl_v_re

# Plot entropy (Fig 3A)
def plot_entropy_boxplots(entropy_f, fig, ax, fig_title=None, week_colors=None, cmap_name='tab20'):
    if 1 in entropy_f.keys():
        pts = entropy_f[1].index.values
        pts0 = entropy_f[0].index.values
        pts_full = np.concatenate((pts, pts0))
    else:
        pts_all = [entropy_f[ii].index.values for ii in entropy_f.keys()]
        pts_full = np.unique(np.concatenate(pts_all))
    # for cl in ['Non-recurrer','Recurrer']
    # for key, val in entropy_f.items():
    rec = 0
    n_rec = 0
    c_rec = []
    c_nrec = []
    c_rec_dict = {}

    if week_colors is None:
        colors = {}
        cmap = matplotlib.cm.get_cmap(cmap_name)
        i = 0
        for ii, t in enumerate(entropy_f.keys()):
            for jj in ['Recurrer', 'Non-recurrer']:
                cv = cmap(i)
                colors[str(t) + jj] = cv
                i += 1
    #         colors = {}
    #         cmap = matplotlib.cm.get_cmap(cmap_name)
    #         i=0
    #         for ii in enumerate(entropy_f.keys()):
    #             colors[ii] = {}
    #             for jj in tlabs[a]:
    #                 cv = cmap(i)
    #                 colors[ii][jj] = cv
    #                 i+=1

    for key, val in entropy_f.items():
        cvec_r = colors[str(key) + 'Recurrer']
        cvec_nr = colors[str(key) + 'Non-recurrer']
        #         if np.round(key) != key:
        #             continue
        if (0, 1) in entropy_f.keys():
            key_n = key[0]
        else:
            key_n = key
        box1 = ax.boxplot(val['Week ' + str(key)].loc[val['Outcome'] == 'Non-recurrer'], positions=[key_n - 0.1],
                          whis=[5, 95], showfliers=False)
        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(box1[element], color=cvec_nr, linewidth=6, alpha=1, zorder=-1)
        #         plt.setp(box1['fliers'], color = False, linewidth = 0, alpha = 1, zorder = -1)
        box2 = ax.boxplot(val['Week ' + str(key)].loc[val['Outcome'] == 'Recurrer'], positions=[key_n + .1],
                          whis=[5, 95], showfliers=False)
        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(box2[element], color=cvec_r, linewidth=6, alpha=1, zorder=-1)

    m1 = {}
    m2 = {}
    for pt in pts_full:
        keys = [key for key in entropy_f.keys() if pt in entropy_f[key].index.values]
        vals = [entropy_f[key]['Week ' + str(key)][pt] for key in keys]
        if 1 in entropy_f.keys():
            if pt in entropy_f[1].index.values:
                df_to_check = entropy_f[1]
            elif pt in entropy_f[0].index.values:
                df_to_check = entropy_f[0]
        else:
            pt_where = [i for i in range(len(pts_all)) if pt in pts_all[i]]
            df_to_check = entropy_f[list(entropy_f.keys())[pt_where[0]]]
            keys = [key[0] for key in keys]

        for key, val in zip(keys, vals):
            if df_to_check['Outcome'][pt] == 'Recurrer':
                cvec_r = colors[str(key) + 'Recurrer']
                c_rec_dict[pt] = keys
                c_rec.extend(keys)
                keys = np.array(keys)
                if rec == key:
                    m1[key], = ax.plot(key + .08 * .05, val, linestyle='none', ms=10,
                                       color=cvec_r, alpha=0.65, label='Recurrer (R)')
                    rec += 1
                #             else:
                # #                 ax.plot(keys+.08, vals, '*',ms = 10,c = 'r', alpha = 0.1)
                ax.scatter(key + .08 + np.random.uniform(-1, 1) * .05, val, marker='s', s=80,
                           color=cvec_r, edgecolors="black", alpha=0.65)
            elif df_to_check['Outcome'][pt] == 'Non-recurrer':
                cvec_nr = colors[str(key) + 'Non-recurrer']
                c_nrec.extend(keys)
                keys = np.array(keys)
                if n_rec == key:
                    m2[key], = ax.plot(key + .08, val, ms=10,
                                       linestyle='none', color=cvec_nr, alpha=0.5, label='Non-recurrer (NR)')
                    n_rec += 1
                #             else:
                #                 ax.plot(keys - .08, vals, '*',ms = 10,c = 'g', alpha = 0.1)
                ax.scatter(key - .08 + np.random.uniform(-1, 1) * .05, val, marker='o', s=80,
                           color=cvec_nr, edgecolors="black", alpha=0.5)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, ymax])

    ax.grid(False, axis='x')
    ax.grid(False, axis='x', which='minor')

    if 1 in entropy_f.keys():
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticks([0, 2.5], minor=True)
        ax.set_xticklabels(labels=['Week 1 to 1.5', 'Week 2 to 2.5', 'Week 3 to 3.5', 'Week 4'],
                           # rotation = 10,  ha = 'center',
                           minor=False)
        ax.set_xticklabels(labels=['\nPre-Antibiotic Treatment', '\nPost-Antibiotic Treatment'],
                           minor=True)
    else:
        ax.set_xticks([1, 2, 3])
        ax.set_xticks([0, 2], minor=True)
        ax.set_xticklabels(labels=['Week 1 to Week 2', 'Week 2 to Week 3', 'Week 3 to Week 4'],
                           # rotation = 10,  ha = 'center',
                           minor=False)
        ax.set_xticklabels(labels=['\nPre-Antibiotic Treatment to Week 1', '\nPost-Antibiotic Treatment'],
                           minor=True)

    ax.set_ylabel('Chao1 Index')
    if fig_title:
        ax.set_title(fig_title)

    markers = {'Non-recurrer': 'o', 'Recurrer': 's'}
    h, l = [], []
    keys = entropy_f.keys()
    for re in ['Recurrer', 'Non-recurrer']:
        h.append(MulticolorPatch([colors[str(key) + re] for key in keys], [markers[re] for key in keys]))
        l.append(re)

    fig.legend(h, l,
               handler_map={MulticolorPatch: MulticolorPatchHandler()})
    return fig, ax

# Plot figure 3B & 3C
def plot_PCA(finalDf, variance, targets, path='pca', target_labels=None, fig_title='',
             colors=None, fig=None, ax=None, cmap_name='tab20', sizes=None, alphas=None,
             markers=None):
    if fig == None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        save_fig = True
    else:
        save_fig = False
    ax.set_xlabel('PC 1, variance explaned= ' +
                  str(np.round(variance[0] * 100, 3)) + '%')
    ax.set_ylabel('PC 2, variance explaned= ' +
                  str(np.round(variance[1] * 100, 3)) + '%')
    ax.set_title(fig_title)

    if sizes is None:
        sizes = {'Recurrer': 90, 'Non-recurrer': 80}
    if alphas is None:
        alphas = {'Recurrer': 0.8, 'Non-recurrer': 0.65}
    if markers is None:
        markers = {'Recurrer': 's', 'Non-recurrer': 'o'}

    if len(targets) == 2 and hasattr(targets[0], "__len__"):
        tlabs = [np.unique(targs).tolist() for targs in targets]
        if len(tlabs[0]) == 2:
            tlabs[0].sort(reverse=True)
            a, b = (0, 1)
        else:
            tlabs[1].sort(reverse=True)
            a, b = (1, 0)

        if colors is None:
            colors = {}
            cmap = matplotlib.cm.get_cmap(cmap_name)
            i = 0
            for ii in tlabs[b]:
                colors[ii] = {}
                for jj in tlabs[a]:
                    cv = cmap(i)
                    colors[ii][jj] = cv
                    i += 1

        all_targs = np.concatenate([list(zip(tlabs[a], x)) for x in itertools.permutations(tlabs[b], len(tlabs[a]))])
        lines = {}
        lines[a] = {}
        lines[b] = {}
        for targ in all_targs:
            indicesToKeep1 = np.where(targets[0] == targ[a])[0]
            # if targ[0] not in lines.keys():
            #     lines[targ[0]] = {}
            indicesToKeep2 = np.where(targets[1] == targ[b])[0]
            ix_combo = list(set(indicesToKeep1).intersection(indicesToKeep2))
            # recur/cleared will always be targ[0]
            # if recur/cleared are first, want recur/cleared to be colors
            # otherwise, want weeks to be colors & recur/cleared to be markers

            # if Recur/cleared is the second label OR if weeks is the second label
            # if targ[b] in tlabs[1]:
            # Condition met for whichever labels we want to be markers
            line = ax.scatter(finalDf.iloc[ix_combo]['PC1'],
                              finalDf.iloc[ix_combo]['PC2'],
                              color=colors[targ[b]][targ[a]], s=sizes[targ[a]], alpha=alphas[targ[a]],
                              marker=markers[targ[a]], edgecolors="gray")

        h = []
        l = []
        for k in tlabs[a]:
            h.append(MulticolorPatch([colors[str(key)][k] for key in tlabs[b]], [markers[k] for key in tlabs[b]]))
            l.append(k)

        leg1 = fig.legend(h, l,
                          handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='right',
                          bbox_to_anchor=(0.85, 0.48),
                          prop={'size': 15})

        h = []
        l = []
        for k in tlabs[b]:
            h.append(MulticolorPatch([colors[k][key] for key in tlabs[a]], [markers[key] for key in tlabs[a]]))
            if k == '0':
                k = 'Pre-Antibiotics'
            elif float(k) < 4:
                k = 'Week ' + k + ' to ' + str(float(k) + 0.5)
            else:
                k = 'Week ' + k
            l.append(k)

        fig.legend(h, l,
                   handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='right', bbox_to_anchor=(1.02, 0.49),
                   prop={'size': 13.9})
        ax.add_artist(leg1)

    else:
        tlabs = np.unique(targets)
        if colors is None:
            colors = {}
            cmap = matplotlib.cm.get_cmap(cmap_name)
            for ii, t in enumerate(tlabs):
                colors[t] = cmap(ii)
        for target, color in zip(tlabs, colors):
            indicesToKeep = np.array(targets == target)
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                       finalDf.loc[indicesToKeep, 'PC2'], c=colors[color], s=50)
        if target_labels is None:
            ax.legend(tlabs, prop={'size': 20})
        else:
            ax.legend(target_labels, prop={'size': 20})
    if save_fig:
        plt.savefig(path + '.pdf')
    else:
        return fig, ax

def pcoa_custom(x, metric = 'braycurtis', metric_mds = True):
    if metric == 'euclidean':
        dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x, metric = 'euclidean'))
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                'PC1', 'PC2'])
        variance = pca.explained_variance_ratio_  # calculate variance ratios
    else:
        if metric == 'spearman':
            corr, p = st.spearmanr(x.T)
            dist_mat = (1 - corr)/2
            dist_mat = (dist_mat + dist_mat.T) / 2
            np.fill_diagonal(dist_mat,0)
        else:
            dist_mat = scipy.spatial.distance.pdist(x, metric = 'braycurtis')
            dist_mat = scipy.spatial.distance.squareform(dist_mat)

        print(dist_mat.shape)
        mds = MDS(n_components = 2, dissimilarity = 'precomputed', metric = metric_mds)
        Y = mds.fit_transform(dist_mat)

        C = np.eye(dist_mat.shape[0]) - (1/dist_mat.shape[0])*np.ones(dist_mat.shape)
        D = dist_mat**2
        B = -0.5*(C@D@C)
        eigen_values, eigen_vectors = np.linalg.eig(B)
        sorted_ixs = np.argsort(eigen_values)[::-1]
        sorted_eig = eigen_values[sorted_ixs]
        E = eigen_vectors[:,sorted_ixs]

        # Y = (E@x)

        variance = [(se/np.sum(abs(sorted_eig))) for se in sorted_eig]
        # variance_tot = np.var(Y,0)
        # variance = (variance_tot/ sum(abs(variance_tot)))*100
        principalDf = pd.DataFrame(np.array(Y)[:,:2], columns = ['PC1', 'PC2'])

    return principalDf, variance, dist_mat


def make_side_heatmap(key, rownames, path_to_save, plot_padj=True):
    df = {}
    for week in [0, 1, 2]:
        if key == '16s' and isinstance(week, list):
            continue
        if key == '16s':
            test = 'deseq2'
            pname = 'padj'
        else:
            test = 'ttest'
            pname = 'BH corrected'
        univ_anal = pd.read_csv(path_to_save + 'univariate_analysis/' + key + '/' + test + '_' + key + str(week) +
                                '.csv', index_col=0)
        #         df[week] = univ_anal.loc[univ_anal[pname]<0.1]
        if plot_padj:
            data_to_plot = univ_anal.iloc[:, [-3, -1]]
        else:
            data_to_plot = univ_anal.iloc[:, -1:]

        not_present = list(set(rownames) - set(data_to_plot.index.values))
        for metab in not_present:
            if plot_padj:
                data_to_plot.loc[len(data_to_plot.index)] = [1, 0]
            else:
                data_to_plot.loc[len(data_to_plot.index)] = [0]
            data_to_plot.index.values[-1] = metab

        df[week] = data_to_plot.loc[rownames]
        if len(rownames) == 0:
            df[week] = data_to_plot

    return df


def get_metab_clustermap(data, key):
    x = data.T
    if key == '16s':
        dist_mat = scipy.spatial.distance.pdist(x, metric='braycurtis')
        dist_mat = scipy.spatial.distance.squareform(dist_mat)

    else:
        corr, p = st.spearmanr(x.T)
        dist_mat = (1 - corr) / 2
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0)

    linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_mat), method='average',
                                              optimal_ordering=True)

    g = sns.clustermap(data.T, row_linkage=linkage, col_cluster=False, row_cluster=True,
                       vmin=-4, vmax=4, cmap=plt.get_cmap('bwr'))

    g.cax.set_visible(False)
    g.fig.set_visible(False)
    return g.dendrogram_row.reordered_ind


def get_pt_clustermap(data, outcomes, key):
    rgb_outcome = {'Recurrer': matplotlib.colors.to_rgba('r')[:-1], 'Non-recurrer': matplotlib.colors.to_rgba('g')[:-1]}
    # pt_outcome = dict(zip(np.unique(pts), generate_colormap(len(np.unique(pts)))))

    o_map = outcomes.map(rgb_outcome)
    df_colors = o_map
    x = data

    if key == '16s':
        dist_mat = scipy.spatial.distance.pdist(x, metric='braycurtis')
        dist_mat = scipy.spatial.distance.squareform(dist_mat)
    else:
        corr, p = st.spearmanr(x.T)
        dist_mat = (1 - corr) / 2
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0)

    linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_mat), method='average',
                                              optimal_ordering=True)

    g = sns.clustermap(data.T, col_linkage=linkage, col_cluster=True, row_cluster=False, col_colors=df_colors,
                       vmin=-4, vmax=4, cmap=plt.get_cmap('bwr'))

    g.cax.set_visible(False)
    g.fig.set_visible(False)
    return g, rgb_outcome

def make_metab_dendrogram(df_dendro, data, fig, ax, custom_dict, skip_offset = 10, buffer = 5):

    ypts_S = [0]
    sub_path_S = []
    coords_S = []
    colors_S = []
    col_dict = {}

    ypts = [0]
    sub_path = []
    hcoords = []
    vcoords = []
    hcolors = []
    vcolors = []
    annotation = []
    order_dict = {}
    order_past = 0
    for pathway,j in custom_dict.items():
        metabs = df_dendro.index.values[df_dendro['SUPER_PATHWAY']==pathway]
        metabs = list(set(metabs).intersection(set(data.columns.values)))
        if len(metabs)<2:
            continue
        txt = pathway.replace(' ', '\n',2)
        if pathway == 'Energy':
            y_add = 20
        else:
            y_add = 15
        sub_pathways = np.unique(df_dendro['SUB_PATHWAY'][df_dendro['SUPER_PATHWAY']==pathway])
        super_gp_size = np.sum([np.sum(df_dendro['SUB_PATHWAY']==p) for p in sub_pathways])
        y_start_sup = ypts_S[-1] + buffer
        y_end_sup = super_gp_size + y_start_sup
        y_mid_sup = (y_end_sup - y_start_sup)/2 + y_start_sup
        xpt = 3

        sub_path_S.append(pathway)
        hline_sup = [[xpt, y_mid_sup],[xpt-2, y_mid_sup]]
        vline_sup = [[xpt-2, y_end_sup],[xpt-2, y_start_sup]]
        hcoords.append(hline_sup)
        vcoords.append(vline_sup)
        hcolors.append('C' + str(j))
        vcolors.append('C' + str(j))
    #     vcolors.extend(['C' + str(j)]*2)

        metabs = df_dendro.index.values[df_dendro['SUPER_PATHWAY']==pathway]
        metabs = list(set(metabs).intersection(set(data.columns.values)))
        order = np.array(get_metab_clustermap(data.loc[:,metabs], 'metabs'))
        order_dict[pathway] = np.array(metabs)[order]
        sub_pathways = np.unique(df_dendro['SUB_PATHWAY'][df_dendro['SUPER_PATHWAY']==pathway])
        annotation.append(((xpt-0.1, y_mid_sup + y_add),txt))

        ypts_S.append(y_end_sup)

    xyvals = np.concatenate(np.concatenate([hcoords, vcoords]))

    line_segments = LineCollection(hcoords, colors = hcolors)
    ax.add_collection(line_segments)
    ax.set_xlim(np.max(xyvals[:,0])+1.1,np.min(xyvals[:,0]))
    ax.set_ylim(np.min(xyvals[:,1])-1, np.max(xyvals[:,1])+1)

    line_segments_v = LineCollection(np.array(vcoords), colors = vcolors, linewidths = 5)
    ax.add_collection(line_segments_v)

    for coord, val in annotation:
        t = ax.annotate(val, coord)
        if coord[0] > 1:
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='gray'))
        else:
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='red'))

    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return fig, ax, order_dict


def plot_heatmap(key, rownames, fig, ax_heat, ax_top, ax_side, dl, figsize=12 * 10, weeks=[0, 1, 2],
                 dtype='data',
                 path_to_save='/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/',
                 cmap_heat='bwr', cmap_sig='PiYG'):
    df_side = make_side_heatmap(key, rownames,
                                path_to_save, plot_padj=True)

    #     if key == '16s':
    #         dat = get_data(key, dtype = dtype)[0]
    #         fin_seqs = list(set(seqs).intersection(set(dat.columns.values)))
    df_all = np.concatenate([df_side[i].iloc[:, -1].values for i in df_side.keys()])
    df_all = df_all[~np.isnan(df_all)]
    for i, week in enumerate(weeks):
        d_week, outcomes = get_data(key, dl, week=week, dtype=dtype, features=rownames)
        if key == '16s':
            data_prop = np.divide(d_week.T, np.sum(d_week, 1)).T
            epsilon = get_epsilon(data_prop)
            geom_means = np.exp(np.mean(np.log(data_prop + epsilon), 1))
            temp = np.divide(data_prop.T, geom_means).T
            epsilon = get_epsilon(temp)
            transformed = np.log(temp + epsilon)
            d_week = standardize(transformed)
        else:
            epsilon = get_epsilon(d_week)
            transformed = np.log(d_week + epsilon)
            d_week = standardize(transformed)
        #         g, rgb_outcome = get_pt_clustermap(d_week, outcomes, key)
        outcome_ixs = np.concatenate([np.where(outcomes.values == 'Recurrer')[0],
                                      np.where(outcomes.values == 'Non-recurrer')[0]])
        ix1 = len(np.where(outcomes.values == 'Recurrer')[0])
        outcome_locs = [ix1 / 2, ix1 + len(np.where(outcomes.values == 'Non-recurrer')[0]) / 2]
        if i == 0:
            outcome_locs[0] = outcome_locs[0] - 5
            outcome_locs[1] = outcome_locs[1] + 5

        if len(rownames) == 0:
            data = d_week.iloc[outcome_ixs, :]
        else:
            #             if key == '16s':
            #                 d_week = d_week.loc[fin_seqs]
            data = d_week.loc[:, rownames].iloc[outcome_ixs, :]

        pos = ax_heat[i].imshow(data.T, cmap=plt.get_cmap(cmap_heat),
                                vmin=-4, vmax=4, interpolation='nearest', aspect='auto');

        ax_heat[i].set_frame_on(False)
        ax_heat[i].axes.get_yaxis().set_visible(False)
        ax_heat[i].axes.get_xaxis().set_ticks([])
        #     ax_metabs[i].axes.get_xaxis().set_visible(False)
        ax_heat[i].grid(False, axis='x')
        if float(week) == 0:
            ax_heat[i].set_xlabel('Pre-antibiotic \n Treatment')
        else:
            ax_heat[i].set_xlabel('Week ' + str(week))

        #         ax_heat[i].set_anchor('E')

        ax_heat[i].xaxis.set_label_position('top')
        from matplotlib.colors import LinearSegmentedColormap
        outcomes = np.expand_dims(outcomes, 0)
        cmap = LinearSegmentedColormap.from_list('r_nr', ['r', 'g'], N=2)
        top = ax_top[i].imshow(np.array(outcomes == 'Non-recurrer')[:, outcome_ixs],
                               cmap=cmap,
                               interpolation='nearest', aspect='auto')
        #         ax_top[i].axes.get_xaxis().set_visible(False)
        ax_top[i].set_xticks(outcome_locs)
        ax_top[i].set_xticklabels(['Recurrer', 'Non-recurrer'], ha='center')
        ax_top[i].tick_params(axis=u'both', which=u'both', length=0)
        ax_top[i].axes.get_yaxis().set_visible(False)
        ax_top[i].set_frame_on(False)

        colors = ['r', 'g']
        for xtick, color in zip(ax_top[i].get_xticklabels(), colors):
            xtick.set_color(color)

        if 't-stat' in df_side[i].columns.values:
            df_side[i]['t-stat'] = df_side[i]['t-stat'].fillna(0)
        else:
            df_side[i].iloc[:, -1:] = df_side[i].iloc[:, -1:].fillna(0)

        vmin = np.min(df_all)
        vmax = np.max(df_all)
        vall = np.max([np.abs(vmin), np.abs(vmax)])
        vmin, vmax = -vall, vall
        right = ax_side[i].imshow(df_side[i].iloc[:, -1:], interpolation='nearest', aspect='auto',
                                  vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap_sig))
        #         ax_side[i].set_anchor('W')

        labels = df_side[i].iloc[:, 0].copy()
        labels[df_side[i].iloc[:, 0] < 0.1] = '*'
        labels[df_side[i].iloc[:, 0] < 0.05] = '**'
        labels[df_side[i].iloc[:, 0] >= 0.1] = ''
        labels[np.isnan(df_side[i].iloc[:, 0])] = ''
        ax_side[i].set_yticks(np.arange(df_side[i].shape[0]))
        ax_side[i].set_yticklabels(labels)
        ax_side[i].tick_params(axis='y', which='major', pad=0.05, left=False, labelleft=False, labelright=True)
        ax_side[i].tick_params(axis=u'both', which=u'both', length=0)
        ax_side[i].yaxis.set_label_position('right')
        ax_side[i].axes.get_xaxis().set_visible(False)
        ax_side[i].set_frame_on(False)

    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.01)

    #     from matplotlib.patches import Patch
    #     handles = [Patch(facecolor=rgb_outcome[name]) for name in rgb_outcome]
    #     legend_1 = ax_top[-1].legend(handles, rgb_outcome,
    #                              bbox_to_anchor=(1, 2), loc='upper left',prop={'size': figsize/12})
    return fig, ax_heat, ax_top, ax_side, right, pos


def lookup_by_names(tree, otu_dict, seq_to_otu, path_to_save):
    sub_gps_to_name = []
    for week in [0, 1, 2]:
        univ_anal = pd.read_csv(path_to_save + 'univariate_analysis/16s/deseq2_16s' + str(week) +
                                '.csv', index_col=0)
        otus = univ_anal.index.values[univ_anal['padj'] < 0.1]
        sub_gps_to_name.extend(otus)
    sub_gps_to_name = np.unique(sub_gps_to_name)
    otu_to_name = [seq_to_otu[seq] for seq in sub_gps_to_name if seq in seq_to_otu.keys()]

    names = {}
    otu_order = []
    for clade in tree.find_clades():
        if clade.branch_length:
            if clade.branch_length > 0.05:
                clade.branch_length = 0.05
        if clade.name:
            if clade.name not in otu_dict.keys():
                dropped = tree.prune(clade)
                continue
            if clade.name in otu_to_name:
                clade.color = 'red'
            otu_order.append(clade.name)
            if isinstance(otu_dict[clade.name]['tax'], str):
                clade.name = otu_dict[clade.name]['tax'] + ' ' + clade.name
            else:
                clade.name = ' '.join(list(otu_dict[clade.name]['tax'].dropna())[5:]) + ', ' + clade.name

            names[clade] = clade.name

    return names, np.unique(otu_order)