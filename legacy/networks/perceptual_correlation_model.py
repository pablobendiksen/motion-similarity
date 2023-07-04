import conf
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn import metrics
import random


def organize_train_test():
    """

    :return: x_train, y_train, x_test, y_test
    x: space, weight, time, flow, dtw
    y: psm
    """
    dfs = pd.read_csv('../csv_files/synthetic_comparisons_dtw.csv')
    dfm = pd.read_csv('../csv_files/mturk_comparisons.csv')


    # map everything bw 0 and 1
    diffs = list(dfs['diff'])
    max = np.max(diffs)
    min = np.min(diffs)
    diffs = (diffs - min) / (max - min)


    space = list(dfm['Space'])
    weight = list(dfm['Weight'])
    time = list(dfm['Time'])
    flow = list(dfm['Flow'])

    x = np.vstack((space, weight, time, flow, diffs))
    x = x.transpose()

    # map everything bw 0 and 1
    y = list(dfm['diff'])  # perceptual output
    max = np.max(y)
    min = np.min(y)
    y = (y - min) / (max - min)


    # Shuffle two ndarrays with same order
    shuffler = np.random.permutation(x.shape[0])
    x_shuffled = x[shuffler]
    y_shuffled = y[shuffler]

    # select random indices
    train_split = (int)(x_shuffled.shape[0] * 0.8)
    x_train = x_shuffled[0:train_split, :]
    y_train = np.array(y_shuffled[0:train_split])


    x_test = x_shuffled[train_split + 1:, :]
    y_test = np.array(y_shuffled[train_split + 1:])


    return x_train, y_train, x_test, y_test

def organize_train_test_by_action(train_actions, test_action):
    """
    :param train_actions: Animations reserved for training
    :param test_action: Animation reserved for testing
    :return:
    """
    dfs = pd.read_csv('../csv_files/synthetic_comparisons_dtw.csv')
    dfm = pd.read_csv('../csv_files/mturk_comparisons.csv')

    # map differences to 0 and 1
    diffs = dfs['diff']
    max = np.max(list(diffs))
    min = np.min(list(diffs))
    dfs['diff'] = (diffs - min) / (max - min)

    # map differences to 0 and 1
    diffm = dfm['diff']
    max = np.max(list(diffm))
    min = np.min(list(diffm))
    dfm['diff'] = (diffm - min) / (max - min)


    test_dfm = dfm[dfm['action'] == test_action]
    test_dfs = dfs[dfs['action'] == test_action]

    space = list(test_dfm['Space'])
    weight = list(test_dfm['Weight'])
    time = list(test_dfm['Time'])
    flow = list(test_dfm['Flow'])

    x_test = np.vstack((space, weight, time, flow, test_dfs['diff']))
    x_test = x_test.transpose()

    y_test = np.array(test_dfm['diff'])  # perceptual output

    # now add train actions

    action_frames_m = []
    action_frames_s = []
    for action in train_actions:
        action_frames_m.append(dfm[dfm['action'] == action])
        action_frames_s.append(dfs[dfs['action'] == action])

    train_dfm = pd.concat(action_frames_m)
    train_dfs = pd.concat(action_frames_s)

    space = list(train_dfm['Space'])
    weight = list(train_dfm['Weight'])
    time = list(train_dfm['Time'])
    flow = list(train_dfm['Flow'])

    x_train = np.vstack((space, weight, time, flow, train_dfs['diff']))
    x_train = x_train.transpose()

    y_train = np.array(train_dfm['diff'])  # perceptual output


    return x_train, y_train, x_test, y_test



def dnn(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential()

    # using a small learning rate provides better accuracy

    opt = Adam(lr=0.001)
    model.add(Dense(12, input_dim=5, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=opt, loss='mse',  metrics=['mse'])

    model.fit(x_train, y_train, epochs=5000, batch_size=16)


    pred= model.predict(x_test)
    # print(np.sqrt(mean_squared_error(y_test,pred)))
    print(np.sqrt(mean_squared_error(y_test, pred)))

    model.save(conf.pdist_model_file)


# dnn(x_train, y_train, x_test, y_test)

def random_forest(x_train, y_train, x_test, y_test):
    regr = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=200)
    regr.fit(x_train, y_train)

    # Make predictions for the test set
    y_pred = regr.predict(x_test)


    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def linear_regression(x_train, y_train, x_test, y_test):
    regr = LinearRegression()

    regr.fit(x_train, y_train)

    # Make predictions for the test set
    y_pred = regr.predict(x_test)



    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




# x_train, y_train, x_test, y_test = organize_train_test()
# x_train, y_train, x_test, y_test = organize_train_test_by_action(["pointing", "picking"], "walking")
x_train, y_train, x_test, y_test = organize_train_test_by_action(["walking", "picking"], "pointing")
# x_train, y_train, x_test, y_test = organize_train_test_by_action(["walking", "pointing"], "picking")

# random_forest(x_train, y_train, x_test, y_test)
# linear_regression(x_train, y_train, x_test, y_test)

dnn(x_train, y_train, x_test, y_test )