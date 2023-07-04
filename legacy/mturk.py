
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multicomp as mc
from statistics import mean
import matplotlib.pyplot as plt
from IPython.display import display
import scipy.stats as sp # for calculating standard error
from bioinfokit.analys import stat
import seaborn as sns
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)

effortList = [[-1, -1, -1, 0], [-1, -1, 1, 0], [-1, 1, -1, 0], [-1, 1, 1, 0],
              [1, -1, -1, 0], [1, -1, 1, 0], [1, 1, -1, 0], [1, 1, 1, 0],
              [-1, -1, 0, -1], [-1, -1, 0, 1] ,[-1, 1 ,0,  -1], [-1, 1, 0, 1],
              [1, -1, 0, -1] ,[1, -1, 0,  1], [1, 1, 0, -1], [1, 1, 0, 1],
              [-1,  0, -1, -1], [-1, 0, -1,  1], [-1, 0, 1,  -1], [-1, 0, 1, 1],
              [1, 0, -1, -1], [1, 0, -1,  1], [1, 0, 1, -1], [1,  0, 1, 1],
              [ 0, -1, -1, -1], [0, -1, -1,  1], [0, -1, 1,  -1], [0, -1, 1, 1],
              [0, 1, -1, -1], [0, 1, -1,  1], [0, 1, 1, -1], [0, 1, 1, 1]]


Space = 0
Time = 1
Weight = 2
Flow = 3

effortName = ['Space', 'Weight', 'Time', 'Flow']


def get_effort(effort_ind, val):
    inds = [i for i, value in enumerate(effortList) if value[effort_ind] == val]
    return inds


def get_drive(url):
    return int(url.split('_')[1].split('.')[0])


def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def get_action(url):
    return url.split('/')[-1].split('_')[0]


def read_csv(name):
    columns = ['Input.video_url', 'Answer.movement-difference.label']
    df = pd.read_csv(name, usecols=columns)
    df = df.rename(columns={'Input.video_url': 'url', 'Answer.movement-difference.label': 'answer'})
    df = df.replace({'Exactly the same': 4, 'Highly similar': 3, 'Moderately similar': 2, 'Slightly similar': 1,
                     'Not similar at all': 0})



    # df_action = df['url'].apply(get_action)
    df['action'] = df['url'].apply(get_action)
    df['drive'] = df['url'].apply(get_drive)
    df = df.drop(['url'], axis=1)

    # Add Effort statistics
    for i in range(4):
        name = effortName[i]
        df[name] = df['drive']

        effort_pos = get_effort(i, 1)
        effort_neg = get_effort(i, -1)
        the_rest = Diff(list(range(0, 32)), (effort_pos + effort_neg))

        ones = np.ones(len(effort_pos))
        neg_ones = np.full(len(effort_neg), -1)
        zeros = np.zeros(len(the_rest))

        df[name] = df[name].replace([effort_pos, effort_neg, the_rest], [ones, neg_ones, zeros])

    return df


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov


def get_effort_stats(df):
    #     formula = 'answer ~ C(Space) + C(Weight) + C(Time) + C(Flow) + C(action) + C(Space):C(action) + C(Weight):C(action) +  C(Time):C(action)+ C(Flow):C(action) '
    # #     formula = 'answer ~ C(Space) + C(Weight) + C(Time) + C(Flow) + C(Space):C(action) + C(Weight):C(action) +  C(Time):C(action)+ C(Flow):C(action) '
    # formula = 'answer ~ C(Space) + C(Time) + C(Weight) + C(Flow) + C(action)'

    # no significant result for groups of 3
    # formula = 'answer ~ Space + Time + Weight + Flow + Space * Weight + Space * Time + Space * Flow + Weight * Time + Weight * Flow + Time  * Flow + Space * Weight * Time  +  Space * Weight * Flow + Space  * Time * Flow + Weight *  Time *  Flow '
    # formula = 'answer ~ C(Space) + C(Time) + C(Weight) + C(Flow) + C(action) + C(Space) * C(Weight) + C(Space) * C(Time) + C(Space) * C(Flow) + C(Weight) * C(Time)  + Weight * C(Flow) + C(Time)  * C(Flow) + C(Space) * C(action)+ C(Weight) * C(action)+ C(Time)  * C(action)+ C(Flow) * C(action)+ C(Space) * C(Time)  * C(action)+ C(Space) * C(Flow)  * C(action)+ C(Weight) * C(Time)  * C(action)+ C(Weight) * C(Flow) * C(action)+ C(Time)   * C(Flow) * C(action)'
    # formula = 'answer ~ Space + Time + Weight + Flow + C(action) + Space * Weight + Space * Time + Space * Flow + Weight * Time + Weight * Flow + Time  * Flow + Space * C(action)+ Weight * C(action)+ Time * C(action)+ Flow * C(action)+ Space * Time * C(action)+ Space * Flow * C(action)+ Weight * Time * C(action)+ Weight * Flow * C(action)+ Time  * Flow * C(action)'


    # formula = 'answer ~ Space * Time * Weight *Flow * C(action) '
    # formula = 'answer ~ Space + Time + Weight + Flow + C(action)'

    formula = 'answer ~ Space + Time + Weight + Flow + Space * Weight + Space * Time + Space * Flow + Weight * Time + Weight * Flow + Time  * Flow '
    #
    #
    # model = ols(formula, df).fit()
    # # print(model.summary().as_latex())
    # aov_table = anova_lm(model, typ=2)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # print(aov_table.to_latex())

    #     eta_squared(aov_table)
    #     omega_squared(aov_table)
    #
    # comp = mc.MultiComparison(df['Space'], df['Time'],  df['Weight'], df['Flow'],df['action'])
    # post_hoc_res = comp.tukeyhsd()
    # print(post_hoc_res.summary())
    #



    res = stat()

    res.anova_stat(df=df, res_var='answer', anova_model=formula)
    print(res.anova_summary)

    #
    res.tukey_hsd(df=df, res_var='answer', xfac_var='Space', anova_model=formula )
    print(res.tukey_summary)

    # # #     perform multiple pairwise comparison (Tukey HSD)
    # m_comp = mc.pairwise_tukeyhsd(endog=df['answer'], groups=df['Flow'], alpha=0.05)
    # print(m_comp)


    # display(aov_table)

    # res = model.resid
    # fig = sm.qqplot(res, line='s')
    # plt.show()

def double_std(array):
 return np.std(array) * 2
def get_drive_stats(df):
    # formula = 'answer ~ C(drive) * C(action)'
    # formula = 'answer ~ C(drive)'
    # formula = 'answer ~ C(action)'
    # model = ols(formula, df).fit()

    # print(model.summary().as_latex())
    # print(model.summary())
    # aov_table = anova_lm(model, typ=2)

    # df2 = df
    # df2['rev_answer'] = 4 - df['answer']

    df_walking = df[df['action'] == "walking"]
    df_pointing = df[df['action'] == "pointing"]
    df_picking = df[df['action'] == "picking"]

    drives = df.groupby('drive').agg([np.mean, np.std, sp.sem])['answer']

    drives_walking = df_walking.groupby('drive').agg([np.mean, np.std, sp.sem])['answer']
    drives_pointing = df_pointing.groupby('drive').agg([np.mean, np.std, sp.sem])['answer']
    drives_picking = df_picking.groupby('drive').agg([np.mean, np.std, sp.sem])['answer']

    # fig = plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(9.5, 5.5)

    sns.boxplot(x="drive", y="answer", data=df, palette="Set3")
    plt.xlabel('Drive index')  ## Label on X axis
    plt.ylabel('Average Drive Similarity')  ##Label on Y axis
    fig.savefig('drives.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    # w = 0.5
    # plt.margins(0, 0)
    # plt.bar(drives.index, drives['mean'], align="center")
    # plt.errorbar(drives.index, drives["mean"], yerr=drives["std"], fmt='o', color='Black', elinewidth=1, capthick=2, errorevery=1, alpha=0.4, ms=2, capsize=2)
    # plt.ylim([0, 4])
    # plt.xticks(range(32), size='small')
    #
    # plt.xlabel('Drive index')  ## Label on X axis
    # plt.ylabel('Average Drive Similarity')  ##Label on Y axis
    #
    # plt.show()
    # fig.savefig('drives.png', dpi=300,  bbox_inches = 'tight',pad_inches = 0)
    # #
    #
    # w = 0.4
    # ax = plt.subplot(111)
    # plt.ylim([0, 4])
    # ax.bar(drives_walking.index*2-w, drives_walking['mean'], width=w, align="center", label ='walking')
    # ax.bar(drives_pointing.index*2, drives_pointing['mean'],width=w, align="center", label ='pointing')
    # ax.bar(drives_picking.index*2+w, drives_picking['mean'],width=w, align="center", label ='picking')
    # ax.legend()
    # plt.errorbar(drives_walking.index*2-w, drives_walking["mean"],  yerr=drives_walking["std"],fmt='o', color='Black', elinewidth=1,capthick=2,errorevery=1, alpha=0.4, ms=1, capsize = 1)
    # plt.errorbar(drives_pointing.index*2, drives_pointing["mean"], yerr=drives_walking["std"], fmt='o', color='Black',elinewidth=1, capthick=2, errorevery=1, alpha=0.4, ms=1, capsize=1)
    # plt.errorbar(drives_picking.index*2 + w, drives_picking["mean"], yerr=drives_walking["std"], fmt='o', color='Black',elinewidth=1, capthick=2, errorevery=1, alpha=0.4, ms=1, capsize=1)
    # #
    # plt.xlabel('Drive index')  ## Label on X axis
    # plt.ylabel('Average Drive Similarity')  ##Label on Y axis
    # #
    # labels = range(32)
    # plt.xticks([i * 2 for i in range(32)], labels)
    # #
    # plt.show()
    # fig.savefig('drivesAll.png', dpi=300, bbox_inches = 'tight',pad_inches = 0)

    # print(drives['mean'])
    N = 153 # sample size
    # t = (np.mean(drives['mean'])- 3.62745098) / np.sqrt((drives['std']*drives['std']/N) +0.572188239*0.572188239/N)
    s = np.sqrt((drives['std']+ 0.572188239) / 2)
    t = (3.6274509 - drives['mean']) / (s * np.sqrt(2 / N))


    df = 2 * N - 2
    p = 1 - stats.t.cdf(t, df=df)

    print("t = " + str(t))
    print("p = " + str(2 * p))

# Format this like the synthetic motion comparison
# drive, action, diff

def get_differences(df):
    # for each drive, compute the average difference

    d = {'drive': [], 'Space': [], 'Weight': [], 'Time': [], 'Flow': [], 'action': [], 'diff': []}
    for anim_name in ['pointing', 'picking', 'walking']:
        for i in range(32):
            vals = df.loc[(df['drive'] == i) ] # & (df['action'] == anim_name) ]
            d['drive'].append(i)
            d['Space'].append(effortList[i][Space])
            d['Weight'].append(effortList[i][Weight])
            d['Time'].append(effortList[i][Time])
            d['Flow'].append(effortList[i][Flow])
            d['action'].append(anim_name)
            d['diff'].append(mean(vals['answer']))

    df_comp = pd.DataFrame(data=d)
    df_comp.to_csv('data/mturk_comparisons.csv')
    print(df_comp)
    return

#
# df = read_csv('data/results.csv')
#
# get_differences(df)

def analyze_correlations(dfs, dfm):

    # write perceptual (mturk) difference in terms of dtw difference
    df = dfs
    t = stats.pearsonr(df['diff'], dfm['diff'])
    print(t)

    # dfs2= dfs[dfs['Flow'] == -1]
    # dfm2 = dfm[dfm['Flow'] == -1]
    #
    # print(dfm2)
    # print(dfs2)
    #
    # # #
    # max = np.max(dfs2['diff'])
    # min = np.min(dfs2['diff'])
    # dfs2['diff'] = (dfs2['diff'] - min) / (max - min)
    #


    # max = np.max(dfm['diff'])
    # min = np.min(dfm['diff'])
    # dfm['diff'] = (dfm['diff'] - min) / (max - min)




    #
    # plt.plot(dfm2['diff'])
    # plt.plot(dfs2['diff'])
    # plt.show()

    df['diffm'] = dfm['diff']


    formula = 'diffm ~ diff *Space * Weight * Time * Flow '
    #
    model = ols(formula, df).fit()

    print(model.summary().as_latex())




# # Linear regression fit
# dfs = pd.read_csv('data/synthetic_comparisons_dtw.csv')
dfs = pd.read_csv('legacy/csv_files/synthetic_comparisons_lcss.csv')
dfm = pd.read_csv('legacy/csv_files/mturk_comparisons.csv')
analyze_correlations(dfs, dfm)

# df = read_csv('data/results.csv')
# get_effort_stats(df)

# get_drive_stats(df)
# get_effort_stats(df)



