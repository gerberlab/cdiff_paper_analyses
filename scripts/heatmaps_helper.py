import pandas as pd
from helper import *
from basic_data_methods_helper import *
from matplotlib.collections import LineCollection


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
        if len(corr.shape) == 0:
            corr = np.array([[1, corr], [corr, 1]])
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

def make_metab_dendrogram(df_dendro, data, fig, ax, custom_order, skip_offset = .10, buffer = 0):

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
    pathways = []
    y_add = 0
    for j,pathway in enumerate(custom_order):
        if isinstance(pathway, list):
            path = pathway[0]
        else:
            path = pathway
        if path in df_dendro['SUB_PATHWAY'].values:
            sup_or_sub = 'SUB_PATHWAY'
        elif path in df_dendro['SUPER_PATHWAY'].values:
            sup_or_sub = 'SUPER_PATHWAY'
        else:
            print(pathway)
            print('ERROR - NO PATHWAY')
            return None
        if isinstance(pathway, list):
            metabs = np.concatenate([df_dendro.index.values[df_dendro[sup_or_sub]==path] for path in pathway])
            pathway = '; '.join(pathway)
        else:
            metabs = df_dendro.index.values[df_dendro[sup_or_sub]==pathway]
        txt = pathway
        metabs = list(set(metabs).intersection(set(data.columns.values)))
        if len(metabs)==0:
            continue
        if len(pathway)>20:
            txt = txt[:20]
        if txt[-1] != 's':
            txt = txt + 's'
    #     sub_pathways = np.unique(df_dendro['SUB_PATHWAY'][df_dendro['SUPER_PATHWAY']==pathway])
        super_gp_size = len(metabs)
        y_start_sup = ypts_S[-1] + buffer
        y_end_sup = super_gp_size + y_start_sup
        y_mid_sup = super_gp_size/2 + y_start_sup
        print(y_mid_sup)
        xpt = 3

        hline_sup = [[xpt, y_mid_sup],[xpt-2, y_mid_sup]]
        vline_sup = [[xpt-2, y_end_sup],[xpt-2, y_start_sup]]
        hcoords.append(hline_sup)
        vcoords.append(vline_sup)
        hcolors.append('C' + str(j))
        vcolors.append('C' + str(j))
    #     vcolors.extend(['C' + str(j)]*2)
        if len(metabs)>1:
            order = np.array(get_metab_clustermap(data.loc[:,metabs], 'metabs'))
            order_dict[pathway] = np.array(metabs)[order]
        else:
            order_dict[pathway] = np.array(metabs)
        pathways.append(pathway)
#         sub_pathways = np.unique(df_dendro[sup_or_sub][df_dendro[sup_or_sub]==pathway])
        annotation.append(((xpt-0.1, y_mid_sup + 0.5),txt))

        ypts_S.append(y_end_sup)

    xyvals = np.concatenate(np.concatenate([hcoords, vcoords]))

    line_segments = LineCollection(hcoords, colors = hcolors)
    ax.add_collection(line_segments)
    ax.set_xlim(np.max(xyvals[:,0]),np.min(xyvals[:,0]))
    ax.set_ylim(np.min(xyvals[:,1]), np.max(xyvals[:,1]))

    line_segments_v = LineCollection(np.array(vcoords), colors = vcolors, linewidths = 5)
    ax.add_collection(line_segments_v)

    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    return fig, ax, order_dict, pathways


def plot_heatmap(key, rownames, fig, ax_heat, ax_side, dl, figsize=12 * 10, weeks=[0, 1, 2],
                 dtype='data',
                 path_to_save='/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/',
                 cmap_heat='vlag', cmap_sig='tab:purple'):
    df_side = make_side_heatmap(key, rownames,
                                path_to_save, plot_padj=True)
    import matplotlib.colors as colors
    df_all = np.concatenate([df_side[i].iloc[:, 0].values for i in df_side.keys()])
    df_all = df_all[~np.isnan(df_all)]
    for i, week in enumerate(weeks):
        d_week, outcomes = get_data(key, dl, week=week, dtype=dtype, features=rownames)
        if key == '16s':
            data_week = np.divide(d_week.T, np.sum(d_week, 1)).T

        else:
            epsilon = get_epsilon(d_week)
            transformed = np.log(d_week + epsilon)
            d_week = standardize(transformed)
        #         g, rgb_outcome = get_pt_clustermap(d_week, outcomes, key)
        outcome_ixs = np.concatenate([np.where(outcomes.values == 'Recurrer')[0],
                                      np.where(outcomes.values == 'Non-recurrer')[0]])

        re_ixs = np.where(outcomes.values == 'Recurrer')[0]
        cl_ixs = np.where(outcomes.values == 'Non-recurrer')[0]
        ix1 = len(np.where(outcomes.values == 'Recurrer')[0])

        if len(rownames) == 0:
            data = d_week.iloc[outcome_ixs, :]
        else:
            data = d_week.loc[:, rownames].iloc[outcome_ixs, :]

        data = data.T
        if key == '16s':
            data.insert(len(re_ixs), 'zeros', -99 * np.ones(data.shape[0]))
        else:
            data.insert(len(re_ixs), 'zeros', 0 * np.ones(data.shape[0]))
        outcome_locs = [ix1 / 2, 1 + ix1 + len(np.where(outcomes.values == 'Non-recurrer')[0]) / 2]
        if i == 0:
            outcome_locs[0] = outcome_locs[0] - 2
            outcome_locs[1] = outcome_locs[1] + 2
        if key != '16s':
            pos = ax_heat[i].imshow(data, cmap=sns.color_palette(cmap_heat, as_cmap=True),
                                    vmin=-4, vmax=4, interpolation='nearest', aspect='auto');
        else:
            pos = ax_heat[i].imshow(data + 1, cmap=sns.light_palette(cmap_heat, as_cmap=True),
                                    interpolation='nearest', aspect='auto',
                                    norm=colors.LogNorm(vmin=1, vmax=data.max().max()));

        ax_heat[i].set_frame_on(False)
        ax_heat[i].axes.get_yaxis().set_visible(False)
        ax_heat[i].grid(False, axis='x')
        if float(week) == 0:
            ax_heat[i].set_xlabel('Pre-antibiotic \n Treatment')
        else:
            ax_heat[i].set_xlabel('Week ' + str(week))
        ax_heat[i].xaxis.set_label_position('bottom')
        outcomes = np.expand_dims(outcomes, 0)
        ax_heat[i].set_xticks(outcome_locs)
        ax_heat[i].xaxis.tick_top()
        if i == 0:
            ax_heat[i].set_xticklabels(['Recurrer       ', 'Non-recurrer'], ha='center')
        else:
            ax_heat[i].set_xticklabels(['Recurrer  ', 'Non-recurrer'], ha='center')
        ax_heat[i].tick_params(axis=u'both', which=u'both', length=0)
        colors2 = ['darkslategray', 'darkcyan']

        if 't-stat' in df_side[i].columns.values:
            df_side[i]['t-stat'] = df_side[i]['t-stat'].fillna(0)
        else:
            df_side[i].iloc[:, -1:] = df_side[i].iloc[:, -1:].fillna(0)

        df_side[i].iloc[:, 0] = df_side[i].iloc[:, 0].fillna(1)

        vmin = np.min(df_all)
        vmax = np.max(df_all)
        print(vmin)
        vmin = 1e-4
        right = ax_side[i].imshow(df_side[i].iloc[:, 0:1], interpolation='nearest', aspect='auto',
                                  vmin=vmin, vmax=1, cmap=sns.light_palette(cmap_sig, reverse=True,
                                                                            as_cmap=True),
                                  norm=colors.LogNorm(vmin=1, vmax=data.max().max()))

        labels = df_side[i].iloc[:, -1].copy()
        labels[(df_side[i].iloc[:, -1] < 0) * (df_side[i].iloc[:, 0] <= 0.05)] = '\u2193'
        labels[(df_side[i].iloc[:, -1] > 0) * (df_side[i].iloc[:, 0] <= 0.05)] = '\u2191'
        labels[df_side[i].iloc[:, -1] == 0] = ''
        labels[df_side[i].iloc[:, 0] > 0.05] = ''
        labels[np.isnan(df_side[i].iloc[:, -1])] = ''
        ax_side[i].set_yticks(np.arange(df_side[i].shape[0]))
        ax_side[i].set_yticklabels(labels)
        ax_side[i].tick_params(axis='y', which='major', pad=0.05, left=False, labelleft=False, labelright=True)
        ax_side[i].tick_params(axis=u'both', which=u'both', length=0)
        ax_side[i].yaxis.set_label_position('right')
        ax_side[i].axes.get_xaxis().set_visible(False)
        ax_side[i].set_frame_on(False)

    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.01)

    return fig, ax_heat, ax_side, right, pos


def lookup_by_names(tree, otu_dict, seq_to_otu, path_to_save, branch_len=0.1):
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
            if clade.branch_length > branch_len:
                clade.branch_length = branch_len
        if clade.name:
            if clade.name not in otu_dict.keys():
                dropped = tree.prune(clade)
                continue
            #             if clade.name in otu_to_name:
            #                 clade.color = 'red'
            otu_order.append(clade.name)
            #             if clade.name == 'ASV_83':
            #                 clade.name = otu_dict[clade.name]['tax'] + ',\n' + clade.name.replace('_','')
            if isinstance(otu_dict[clade.name]['tax'], str):
                if ' ' in otu_dict[clade.name]['tax']:
                    clade.name = '*' + otu_dict[clade.name]['tax'].replace(
                        '_', ' ') + ', ' + clade.name.replace('_', ' ')
                else:
                    clade.name = '**' + otu_dict[clade.name]['tax'].replace(
                        '_', ' ') + ', ' + clade.name.replace('_', ' ')
            else:
                clade.name = ' '.join(list(otu_dict[clade.name]['tax'].dropna())[5:]).replace(
                    '_', '') + ', ' + clade.name.replace('_', ' ')

            if clade.name == '**, ASV 89' or clade.name == '**, ASV 204':
                clade.name = '***Lachnospiraceae' + clade.name.split('**')[-1]
            if clade.name == '**, ASV 132':
                clade.name = '***Ruminococcaceae' + clade.name.split('**')[-1]

            names[clade] = clade.name

    return names


def get_rownames(rownames):
    slab = []
    letter_len = 30
    for r in rownames:
        if '(' in r:
            r = r.split(' (')[0]
        r = r.replace('*','')
        r = r.replace('beta','\u03B2')
        xr = r.split('\n')
        while any([len(xxr)>letter_len for xxr in xr]):
            if ' ' in r and '\n' not in r:
                tmp_spl = r.split(' ')
                out = []
                for tmp in tmp_spl:
                    if len(tmp.split('\n')[-1])>=int(letter_len/2):
                        out.append(tmp + '\n')
                    else:
                        out.append(tmp + ' ')
                rtemp  = ''.join(out)
                if rtemp[-1]==' ':
                    rtemp = rtemp[:-1]
                xr = r.split('\n')
                if rtemp != r:
                    r = rtemp
                    continue
            if '-' in r and '-\n' not in r:
                tmp_spl = r.split('-')
                out = []
                for tmp in tmp_spl[:-1]:
                    if len(tmp.split('\n')[-1])>=int(letter_len/2):
                        out.append(tmp + '-\n')
                    else:
                        out.append(tmp + '-')
                out.append(tmp_spl[-1])
                rtemp  = ''.join(out)
                xr = r.split('\n')
                if rtemp != r:
                    r = rtemp
                    continue
            rr = r.split('\n')
            rout = []
            for ii,ri in enumerate(rr):
                if len(ri)>=letter_len:
                    ri = ri[:int(len(ri)/2)] + '-\n' + ri[int(len(ri)/2):]
                if ii != len(rr):
                    rout.append(ri + '\n')
                else:
                    rout.append(ri)
            r = ''.join(rout)
            xr = r.split('\n')
        r = r.replace('\n\n','')
        if r[-1] == '\n':
            r = r[:-1]
        slab.append(r)
    return slab