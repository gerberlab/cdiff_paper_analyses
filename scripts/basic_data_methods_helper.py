import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import multipletests
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import matplotlib
import seaborn as sns
import itertools
from sklearn.manifold import MDS
import scipy
from sklearn.decomposition import PCA
from helper import *

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors, shapes, alphas=None):
        self.colors = colors
        self.shapes = shapes
        if alphas:
            self.alphas = alphas
        else:
            if isinstance(self.colors, list):
                self.alphas = [1] * len(self.colors)
            else:
                self.alphas = 1


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.width = len(orig_handle.colors) * handlebox.height + handlebox.height * 0.1 * len(orig_handle.colors)
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            if orig_handle.shapes[i] == 's':
                patches.append(Rectangle([width / len(orig_handle.colors) * i + (handlebox.height * 0.1) * i, 0],
                                         height,
                                         height,
                                         facecolor=c,
                                         alpha=orig_handle.alphas[i],
                                         edgecolor='gray'))
            else:
                patches.append(Ellipse([(width / len(orig_handle.colors) * i) + width / len(orig_handle.colors) / 2 +
                                        (handlebox.height * 0.1) * i, height / 2],
                                       height,
                                       height,
                                       facecolor=c,
                                       alpha=orig_handle.alphas[i],
                                       edgecolor='gray'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch

def set_font_sizes(figsize, ftype_dict = {'font': 8, 'axes_title': 16, 'ax5s_label': 12,
                                          'xtick_label': 10, 'ytick_label': 10, 'legend': 10, 'figure_title': 18}):
    default_a = (6*8)/7
    if figsize == None:
        area = default_a
    else:
        area = figsize[0]*figsize[1]/((figsize[0] + figsize[1])/2)
    for ftype, factor in ftype_dict.items():
        size = factor*area/default_a
        if 'title' in ftype:
            plt.rc(ftype.split('_')[0], titlesize=size)
        elif 'label' in ftype:
            plt.rc(ftype.split('_')[0], labelsize=size)
        elif 'legend' in ftype:
            plt.rc(ftype.split('_')[0], fontsize=size)
        else:
            plt.rc(ftype.split('_')[0], size=size)
        print(ftype + ': ' + str(size))

def get_panal_locs(ax):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    return (xmin,ymax + 0.1*(ymax-ymin))

def entropy_over_time(entropy_df, tmpts = [0,1,2], use_ranksum = False, use_one_sided = True):
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
    return df_over_time

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

    df_cl_v_re = pd.DataFrame({'P-value': pval, 'FDR': pval_corr})
    return df_cl_v_re


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
    else:
        colors = {}
        for ii, t in enumerate(entropy_f.keys()):
            for jj in ['Recurrer', 'Non-recurrer']:
                colors[str(t) + jj] = week_colors[jj]

    for key, val in entropy_f.items():
        cvec_r = colors[str(key) + 'Recurrer']
        cvec_nr = colors[str(key) + 'Non-recurrer']
        #         if np.round(key) != key:
        #             continue
        if (0, 1) in entropy_f.keys():
            key_n = key[0]
        else:
            key_n = key
        box1 = ax.boxplot(val['Week ' + str(key)].loc[val['Outcome'] == 'Non-recurrer'], positions=[key_n - 0.18],
                          whis=[5, 95], showfliers=False, widths = 0.3)
        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(box1[element], color=cvec_nr, linewidth=2, alpha=1, zorder=-1)
        #         plt.setp(box1['fliers'], color = False, linewidth = 0, alpha = 1, zorder = -1)
        box2 = ax.boxplot(val['Week ' + str(key)].loc[val['Outcome'] == 'Recurrer'], positions=[key_n + .18],
                          whis=[5, 95], showfliers=False, widths = 0.3)
        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(box2[element], color=cvec_r, linewidth=2, alpha=1, zorder=-1)

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
                #             else:
                # #                 ax.plot(keys+.08, vals, '*',ms = 10,c = 'r', alpha = 0.1)
                ax.scatter(key + .18 + np.random.uniform(-1, 1) * .05, val, marker='o', s=20,
                           color=cvec_r, alpha=0.5)
            elif df_to_check['Outcome'][pt] == 'Non-recurrer':
                cvec_nr = colors[str(key) + 'Non-recurrer']
                c_nrec.extend(keys)
                keys = np.array(keys)
                #             else:
                #                 ax.plot(keys - .08, vals, '*',ms = 10,c = 'g', alpha = 0.1)
                ax.scatter(key - .18 + np.random.uniform(-1, 1) * .05, val, marker='o', s=20,
                           color=cvec_nr, alpha=0.5, zorder=3)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, ymax])

    ax.grid(False, axis='x')
    ax.grid(False, axis='x', which='minor')

    if 1 in entropy_f.keys():
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticks([0], minor=True)
        ax.set_xticklabels(labels=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                           # rotation = 10,  ha = 'center',
                           minor=False)
        ax.set_xticklabels(labels=['Pre-Antibiotic\nTreatment'],
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

    markers = {'Non-recurrer': 'o', 'Recurrer': 'o'}
    h, l = [], []
    keys = entropy_f.keys()
    if week_colors is None:
        for re_lab, re in zip(['R','NR'],['Recurrer', 'Non-recurrer']):
            h.append(MulticolorPatch([colors[str(key) + re] for key in keys], [markers[re] for key in keys]))
            l.append(re_lab)

        fig.legend(h, l,
                   handler_map={MulticolorPatch: MulticolorPatchHandler()})
    else:
        a1 = ax.scatter([], [], marker='o', s=50,
                        color=week_colors['Recurrer'], alpha=0.65)
        a2 = ax.scatter([], [], marker='o', s=50,
                        color=week_colors['Non-recurrer'], alpha=0.65)

        ax.legend([a1, a2], ['R', 'NR'], loc='upper right')
    return fig, ax


def plot_PCA(finalDf, variance, targets, path='pca', target_labels=None, fig_title='',
             colors=None, fig=None, ax=None, cmap_name='tab10', sizes=None, alphas=None,
             markers=None, legend=False, x_label = 'PC 1, variance explained=',
             y_label = 'PC 2, variance explained='):
    if fig == None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        save_fig = True
    else:
        save_fig = False
    ax.set_xlabel(x_label +
                  str(np.round(variance[0] * 100, 3)) + '%')
    ax.set_ylabel(y_label +
                  str(np.round(variance[1] * 100, 3)) + '%')
    ax.set_title(fig_title)

    if sizes is None:
        sizes = {'Recurrer': 30, 'Non-recurrer': 20}
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
                    cv = cmap(i + 10)
                    colors[ii][jj] = cv
                    i += 1

        all_targs = np.concatenate([list(zip(tlabs[a], x)) for x in itertools.permutations(tlabs[b], len(tlabs[a]))])
        lines = {}
        lines[a] = {}
        lines[b] = {}
        for targ in all_targs:
            indicesToKeep1 = np.where(targets[0] == targ[a])[0]
            indicesToKeep2 = np.where(targets[1] == targ[b])[0]
            ix_combo = list(set(indicesToKeep1).intersection(indicesToKeep2))
            line = ax.scatter(finalDf.iloc[ix_combo]['PC1'],
                              finalDf.iloc[ix_combo]['PC2'],
                              color=colors[targ[b]][targ[a]], s=sizes[targ[a]], alpha=alphas[targ[a]],
                              marker=markers[targ[a]], edgecolors="gray")

        h = []
        l = []
        for k in tlabs[a]:
            h.append(MulticolorPatch([colors[str(key)][k] for key in tlabs[b]], [markers[k] for key in tlabs[b]]))
            l.append(k)

        #         leg1 = fig.legend(h, l,
        #                    handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc = 'right', bbox_to_anchor=(1.2, .8),
        #                          prop={'size': 10})
        legend_1 = (h, l)

        h = []
        l = []
        for k in tlabs[b]:
            h.append(MulticolorPatch([colors[k][key] for key in tlabs[a]], [markers[key] for key in tlabs[a]]))
            if k == '0':
                k = 'Pre-Antibiotics'
            elif float(k) < 4:
                k = 'Week ' + k
                # + ' to ' + str(float(k)+0.5)
            else:
                k = 'Week ' + k
            l.append(k)
        legend_2 = (h, l)
    #         leg2 = fig.legend(h, l,
    #            handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc = 'right', bbox_to_anchor=(1.18, 0.55),
    #                    prop={'size': 10})
    #         ax.add_artist(leg1)

    else:
        legend_2 = None
        legend_1 = None
        tlabs = np.unique(targets)
        if colors is None:
            colors = {}
            cmap = sns.color_palette(cmap_name, as_cmap=True)
            for ii, t in enumerate(tlabs):
                colors[t] = cmap(ii + 4)
        for target, color in zip(tlabs, colors):
            indicesToKeep = np.array(targets == target)
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                       finalDf.loc[indicesToKeep, 'PC2'], c=colors[color], s=25)

        if legend:
            if target_labels is None:
                ax.legend(tlabs, prop={'size': 10})
            else:
                ax.legend(target_labels, prop={'size': 10})
    if save_fig:
        plt.savefig(path + '.pdf')
    else:
        return fig, ax, legend_2, legend_1

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


def univariate_w_chi2(x, targets, method = 'ranksum', cutoff=.05, chi_sqr = False):
    if isinstance(targets[0], str):
        targets = (np.array(targets) == 'Recurrer').astype('float')
    else:
        targets = np.array(targets)
    pval = []
    teststat = []
    methods = []
    direction = []
    for i in range(x.shape[1]):

        if (len(np.unique(x.iloc[:,i]))<=3 or (x.iloc[:,i].dtypes!=np.int and x.iloc[:,i].dtypes!=np.float64)) and chi_sqr:
            xcat = x.iloc[:,i]
            vals = np.unique(xcat)
            tab = pd.crosstab(index = xcat, columns = targets)
            if tab.shape[0]==1:
                p = 1
                s = 1
                met_out = 'fisher exact'
                direction.append(0)
            elif len(np.unique(x.iloc[:,i]))>2:
                f1 = tab[1].tolist()
                prior = np.sum(targets==1)/len(targets)
                f0 = [np.sum(xcat==xc)*prior for xc in np.unique(xcat)]
                s, p = st.chisquare(f1, f0)
                dd = []
                for i in np.arange(len(f1)):
                    if f1[i]<f0[i]:
                        dd.append(-1)
                    elif f1[i]>f0[i]:
                        dd.append(1)
                    else:
                        dd.append(0)
                direction.append(dd)
                met_out = 'chi-squared'
            elif len(np.unique(x.iloc[:,i]))==2:
                s, p = st.fisher_exact(tab)
                if s < 1:
                    direction.append(-1)
                elif s > 1:
                    direction.append(1)
                else:
                    direction.append(0)
                met_out = 'fisher-exact'
        else:
            met_out = method
            xin = np.array(x)[:, i]
            X = xin[targets == 1]
            Y = xin[targets == 0]
            # xin1 = (xin - np.min(xin,0))/(np.max(xin,0)-np.min(xin,0))
            if met_out == 'ranksum':
                s, p = st.ranksums(X, Y)
            elif met_out == 'kruskal':
                try:
                    s, p = st.kruskal(X,Y)
                except:
                    p = 1
            elif met_out == 'ttest':
                s, p = st.ttest_ind(X,Y)

            if s < 0:
                direction.append(-1)
            elif s > 0:
                direction.append(1)
            else:
                direction.append(0)
        pval.append(p)
        teststat.append(s)
        methods.append(met_out)

    pval = np.array(pval)
    pval[np.isnan(pval)] = 1
    # corrected, alpha = bh_corr(np.array(pval), .05)

    reject, corrected, a1, a2 = multipletests(pval, alpha=.05, method='fdr_bh')
    df = pd.DataFrame(np.vstack((pval, corrected, teststat, direction, methods)).T, columns=[
        'P_Val', 'BH corrected', 'test statistic','direction','method'], index=x.columns.values)
    return df.sort_values('P_Val', ascending=True)