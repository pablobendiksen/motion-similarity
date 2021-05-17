import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multicomp as mc
import conf

# import tensorflow as tf
import organize_synthetic_data as osd
from tslearn.metrics import soft_dtw
from tslearn.metrics import lcss

import numpy as np


effortList = [[-1, -1, -1, 0], [-1, -1, 1, 0], [-1, 1, -1, 0], [-1, 1, 1, 0], [1, -1, -1, 0], [1, -1, 1, 0], [1, 1, -1, 0], [1, 1, 1, 0],
                                       [-1, -1, 0, -1], [-1, -1, 0, 1],[-1, 1,0,  -1], [-1, 1, 0, 1], [1, -1, 0, -1],[1, -1, 0,  1], [1, 1, 0, -1], [1, 1, 0, 1],
                                        [-1,  0, -1, -1], [-1, 0, -1,  1], [-1, 0, 1,  -1], [-1, 0, 1, 1], [1, 0, -1, -1], [1, 0, -1,  1], [1, 0, 1, -1], [1,  0, 1, 1],
                                       [ 0, -1, -1, -1], [0, -1, -1,  1], [0, -1, 1,  -1], [0, -1, 1, 1], [0, 1, -1, -1], [0, 1, -1,  1], [0, 1, 1, -1], [0, 1, 1, 1]]

Space = 0
Time = 1
Weight = 2
Flow = 3


effortName = ['space', 'time', 'weight', 'flow']

def get_effort(effort_ind, val):
    inds = [i for i, value in enumerate(effortList) if value[effort_ind] == val]
    return inds

anim_name = "pointing"

# dtw_df = {'pointing': [], 'picking': [], 'walking':[]}

def compute_distance_for_all(func=soft_dtw):
    d = {'drive': [], 'Space': [], 'Weight': [], 'Time': [], 'Flow': [], 'action': [], 'diff': []}

    for anim_name in ['pointing', 'picking', 'walking']:
        neutral = osd.load_effort_animation(anim_name, [0, 0, 0, 0])
        for i in range(32):
            # Drives
            drive = effortList[i]
            compared = osd.load_effort_animation(anim_name, drive)
            diff = func(neutral, compared, eps=15)
            d['action'].append(anim_name)
            d['drive'].append(i)
            d['diff'].append(diff)

            d['Space'].append(effortList[i][Space])
            d['Weight'].append(effortList[i][Weight])
            d['Time'].append(effortList[i][Time])
            d['Flow'].append(effortList[i][Flow])

    # write to a csv file
    dtw_dt = pd.DataFrame(data=d)
    dtw_dt.to_csv('data/synthetic_comparisons_lcss.csv')
    return dtw_dt


def get_drive_stats(df):
    formula = 'diff ~ C(drive)'
    model = ols(formula, df).fit()

    # print(model.summary().as_latex())
    print(model.summary())
    aov_table = anova_lm(model, typ=2)


    # display(aov_table)


# compute_distance_for_all(lcss)

compute_distance_for_all(soft_dtw)



# Find correlations between synthetic distances and mturk distances

# get_drive_stats(df)

