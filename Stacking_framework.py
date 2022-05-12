from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN as SimpleRNN, RNN, LSTM, GRU, Dropout, Dense, Bidirectional
import pandas as pd
import numpy as np
import warnings
from IPython.core.debugger import Tracer
import imblearn as im
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # to Ignore the deprication Warnings


def create_windows(data, window_shape, step=1, start_id=None, end_id=None):
    """the function for converting the 2-D data to 3-D data for RNN based Networks"""
    data = np.asarray(data)
    data = data.reshape(-1, 1) if np.prod(data.shape) == max(data.shape) else data

    start_id = 0 if start_id is None else start_id
    end_id = data.shape[0] if end_id is None else end_id

    data = data[int(start_id):int(end_id), :]
    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim
    slices = tuple(slice(None, None, st) for st in step)
    indexing_strides = data[slices].strides
    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))

    window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)

    return np.squeeze(window_data, 1)


def absolute_maximum_scale(series):
    """A function for  Normalize the data"""
    return series / series.abs().max()


def rnn_model(hidden_units, input_shape):
    """Creating the Recurrent Neural Network"""
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, return_sequences=True, ))
    model.add(SimpleRNN(hidden_units, return_sequences=True))
    model.add(SimpleRNN(hidden_units, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.build()
    return model


def brnn_model(hidden_units, input_shape):
    """Creating Bidirectional RNN"""
    model = Sequential()
    model.add(Bidirectional(
        SimpleRNN(hidden_units, input_shape=input_shape, recurrent_dropout=0.3, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(SimpleRNN(hidden_units, recurrent_dropout=0.3, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(SimpleRNN(hidden_units, recurrent_dropout=0.3, return_sequences=False, dropout=0.3)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def lstm_model(hidden_units, input_shape):
    """Ceating LSTM """
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape, recurrent_dropout=0.3, return_sequences=True, dropout=0.3))
    model.add(LSTM(hidden_units, recurrent_dropout=0.3, return_sequences=True, dropout=0.3))
    model.add(LSTM(hidden_units, recurrent_dropout=0.3, return_sequences=False, dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def gru_model(hidden_units, input_shape):
    """Creating the GRU"""
    model = Sequential()
    model.add(GRU(hidden_units, input_shape=input_shape, recurrent_dropout=0.3, return_sequences=True, dropout=0.3))
    model.add(GRU(hidden_units, recurrent_dropout=0.3, return_sequences=True, dropout=0.3))
    model.add(GRU(hidden_units, recurrent_dropout=0.3, return_sequences=False, dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


data = pd.read_excel("nasdaq_full.xlsx")
data = data[0:4054]
X = data.loc[:, data.columns != 'label']  # dividing the data to feature matrix
Y = data.loc[:, data.columns == 'label']  # labels of data
X = X.drop(columns=['Unnamed: 0', 'Date', 'Open', 'High', 'Low', 'Close'], axis=1)  # eliminating Useless Features
for col in X.columns:
    X[col] = absolute_maximum_scale(X[col])

train_set_x = X[0:2400]  # first 2400 records separated for training purposes
train_set_x.reset_index(inplace=True, drop=True)
train_set_y = Y[0:2400]
train_set_y.reset_index(inplace=True, drop=True)

"""sm= im.under_sampling.RandomUnderSampler(sampling_strategy='majority')

train_set_x, train_set_y = sm.fit_resample(train_set_x, train_set_y)"""

test_set_x = X[2400:]  # rest of the data for final test phase
test_set_x.reset_index(inplace=True, drop=True)
test_set_y = Y[2400:]
test_set_y.reset_index(inplace=True, drop=True)

kf8 = StratifiedKFold(n_splits=8, shuffle=False)  # specifying the fold size

# Random forest generating class
# parameters for tuning : n_estimator (50, 100, 150), Max_features (2, 13, 24)
random_forest = RandomForestClassifier(n_estimators=100, max_features=2)
# Extremely randomized trees (ERT)
# parameters for tuning : n_estimator (50, 100, 150), Max_features (2, 13, 24)
extreme_random_tree = ExtraTreesClassifier(n_estimators=150, max_features=2)
# XGBoost
# parameters for tuning : colsample_bytree (0.6, 0.8, 1.0) , learning_rate (0.05, 0.1, 0.15)
xg_boost = xgb.XGBClassifier(colsample_bytree=0.8, learning_rate=0.05)
# LightGBM
# parameters for tuning : n_estimators (50, 100, 150), reg_alpha (0.01, 0.1, 1), reg_lambda (0.01, 0.1, 1)
light_gbm = LGBMClassifier(n_estimators=50, reg_alpha=0.01, reg_lambda=0.1)

#  RNN
rnn = rnn_model(hidden_units=128, input_shape=(1, 23))

# BRNN
brnn = brnn_model(hidden_units=128, input_shape=(1, 23))

# LSTM
lstm = lstm_model(hidden_units=128, input_shape=(1, 23))

# GRU
gru = gru_model(hidden_units=128, input_shape=(1, 23))

logistic_model = LogisticRegression(penalty="l1", C=0.1, solver="liblinear")  # , C=0.1, solver="liblinear"

level2_dataset = pd.DataFrame(columns=['RF', 'ERT', 'XGB', 'LGBM', 'RNN', 'BRNN', 'LSTM', 'GRU', 'label'])  # new
# DataFrae to for data for next level classifier (Meta Classifier)
look_back = 1
look_ahead = 1

# evaluation phase
for train_index, validation_index in kf8.split(train_set_x, train_set_y):  # using train and validation set in CV
    x_train = train_set_x.iloc[train_index]  # training set , provided by CV
    y_train = train_set_y.iloc[train_index]

    x_valid = train_set_x.iloc[validation_index]  # validation set provided by CV
    y_valid = train_set_y.loc[validation_index]

    # fitting the base level classifiers with training set
    random_forest.fit(x_train, y_train.values.ravel())
    extreme_random_tree.fit(x_train, y_train.values.ravel())
    xg_boost.fit(x_train, y_train.values.ravel())
    light_gbm.fit(x_train, y_train.values.ravel())

    # extracting the prediction probability on validation set
    random_forest_predict = random_forest.predict_proba(x_valid)
    print("RF : ", random_forest.score(x_valid, y_valid.values.ravel()))

    extreme_random_tree_predict = extreme_random_tree.predict_proba(x_valid)
    print("ERT : ", extreme_random_tree.score(x_valid, y_valid.values.ravel()))

    xg_boost_predict = xg_boost.predict_proba(x_valid)
    print("XGBoost : ", xg_boost.score(x_valid, y_valid.values.ravel()))

    light_gbm_predict = light_gbm.predict_proba(x_valid)
    print("LightGBM : ", light_gbm.score(x_valid, y_valid.values.ravel()))

    # Converting the trainig data set to be used in the RNN based Networks
    x_train_np = x_train.to_numpy()
    x_train_np = create_windows(x_train_np, window_shape=look_back, end_id=-look_ahead)
    y_train_np = y_train.to_numpy()
    y_train_np = create_windows(y_train_np, window_shape=look_back, end_id=-look_ahead)

    x_valid_np = x_valid.to_numpy()
    x_valid_np = create_windows(x_valid_np, window_shape=look_back, end_id=-look_ahead)
    y_valid_np = y_valid.to_numpy()
    y_valid_np = create_windows(y_valid_np, window_shape=look_back, end_id=-look_ahead)

    # fitting the RNN based networks
    rnn.fit(x_train_np, y_train_np, validation_data=(x_valid_np, y_valid_np))
    RNN_predict = rnn.predict(x_valid_np)

    brnn.fit(x_train_np, y_train_np, validation_data=(x_valid_np, y_valid_np))
    BRNN_predict = brnn.predict(x_valid_np)

    lstm.fit(x_train_np, y_train_np, validation_data=(x_valid_np, y_valid_np))
    LSTM_predict = lstm.predict(x_valid_np)

    gru.fit(x_train_np, y_train_np, validation_data=(x_valid_np, y_valid_np))
    GRU_predict = gru.predict(x_valid_np)

    # using the produced prediction probabilities to form news data set for Meta classifier
    for i in range(0, y_valid.shape[0]):

        if level2_dataset.empty:  # check if the new dataset is empty or not
            if y_valid.iloc[0, 0] == 0:  # Check if true label is 0 or not, if it is zero we use the first column of
                # our produced prediction probability
                level2_dataset.at[0, 'RF'] = random_forest_predict[0, 0]
                level2_dataset.at[0, 'ERT'] = extreme_random_tree_predict[0, 0]
                level2_dataset.at[0, 'XGB'] = xg_boost_predict[0, 0]
                level2_dataset.at[0, 'LGBM'] = light_gbm_predict[0, 0]
                level2_dataset.at[0, 'label'] = y_valid.iloc[0, 0]

                level2_dataset.at[0, 'RNN'] = None
                level2_dataset.at[0, 'BRNN'] = None
                level2_dataset.at[0, 'LSTM'] = None
                level2_dataset.at[0, 'GRU'] = None
            else:
                level2_dataset.at[0, 'RF'] = random_forest_predict[0, 1]
                level2_dataset.at[0, 'ERT'] = extreme_random_tree_predict[0, 1]
                level2_dataset.at[0, 'XGB'] = xg_boost_predict[0, 1]
                level2_dataset.at[0, 'LGBM'] = light_gbm_predict[0, 1]
                level2_dataset.at[0, 'label'] = y_valid.iloc[0, 0]

                level2_dataset.at[0, 'RNN'] = None
                level2_dataset.at[0, 'BRNN'] = None
                level2_dataset.at[0, 'LSTM'] = None
                level2_dataset.at[0, 'GRU'] = None

        else:  # if the new DataFrame is not empty we should find the last index to use attaching new data

            level2_dataset.reset_index(inplace=True, drop=True)
            level2_dataset_ind = level2_dataset.index[-1]

            if y_valid.iloc[i, 0] == 0:  # Check if true label is 0 or not, if it is zero we use the first column of
                # our produced prediction probability
                if i == 0:
                    level2_dataset_ind += 1
                    level2_dataset.at[level2_dataset_ind + i, 'RF'] = random_forest_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'ERT'] = extreme_random_tree_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'XGB'] = xg_boost_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'LGBM'] = light_gbm_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'label'] = y_valid.iloc[i, 0]

                    level2_dataset.at[level2_dataset_ind + i, 'RNN'] = None
                    level2_dataset.at[level2_dataset_ind + i, 'BRNN'] = None
                    level2_dataset.at[level2_dataset_ind + i, 'LSTM'] = None
                    level2_dataset.at[level2_dataset_ind + i, 'GRU'] = None

                else:
                    level2_dataset.at[level2_dataset_ind + i, 'RF'] = random_forest_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'ERT'] = extreme_random_tree_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'XGB'] = xg_boost_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'LGBM'] = light_gbm_predict[i, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'label'] = y_valid.iloc[i, 0]

                    level2_dataset.at[level2_dataset_ind + i, 'RNN'] = RNN_predict[i - 1, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'BRNN'] = BRNN_predict[i - 1, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'LSTM'] = LSTM_predict[i - 1, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'GRU'] = GRU_predict[i - 1, 0]
            else:
                if i == 0:
                    level2_dataset_ind += 1
                    level2_dataset.at[level2_dataset_ind + i, 'RF'] = random_forest_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'ERT'] = extreme_random_tree_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'XGB'] = xg_boost_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'LGBM'] = light_gbm_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'label'] = y_valid.iloc[i, 0]

                    level2_dataset.at[level2_dataset_ind + i, 'RNN'] = None
                    level2_dataset.at[level2_dataset_ind + i, 'BRNN'] = None
                    level2_dataset.at[level2_dataset_ind + i, 'LSTM'] = None
                    level2_dataset.at[level2_dataset_ind + i, 'GRU'] = None

                else:
                    level2_dataset.at[level2_dataset_ind + i, 'RF'] = random_forest_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'ERT'] = extreme_random_tree_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'XGB'] = xg_boost_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'LGBM'] = light_gbm_predict[i, 1]
                    level2_dataset.at[level2_dataset_ind + i, 'label'] = y_valid.iloc[i, 0]

                    level2_dataset.at[level2_dataset_ind + i, 'RNN'] = RNN_predict[i - 1, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'BRNN'] = BRNN_predict[i - 1, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'LSTM'] = LSTM_predict[i - 1, 0]
                    level2_dataset.at[level2_dataset_ind + i, 'GRU'] = GRU_predict[i - 1, 0]

level2_dataset = level2_dataset.dropna()  # Dropping the NA values.
level2_dataset.reset_index(inplace=True, drop=True)  # reindexing the new DataFrame

level2_x = level2_dataset.loc[:, level2_dataset.columns != 'label']
level2_y = level2_dataset.loc[:, level2_dataset.columns == 'label']
level2_y = level2_y.astype('int')

logistic_model.fit(level2_x, level2_y.values.ravel())
print("score: ", logistic_model.score(level2_x, level2_y.values.ravel()))

# final train phase
# fitting the base level classifiers with training set
# extracting the prediction probability on validation set

random_forest.fit(train_set_x, train_set_y.values.ravel())

extreme_random_tree.fit(train_set_x, train_set_y.values.ravel())

xg_boost.fit(train_set_x, train_set_y.values.ravel())

light_gbm.fit(train_set_x, train_set_y.values.ravel())

random_forest_predict = random_forest.predict_proba(train_set_x)
print("RF : ", random_forest.score(train_set_x, train_set_y.values.ravel()))

extreme_random_tree_predict = extreme_random_tree.predict_proba(train_set_x)
print("ERT : ", extreme_random_tree.score(train_set_x, train_set_y.values.ravel()))

xg_boost_predict = xg_boost.predict_proba(train_set_x)
print("XGBoost : ", xg_boost.score(train_set_x, train_set_y.values.ravel()))

light_gbm_predict = light_gbm.predict_proba(train_set_x)
print("LightGBM : ", light_gbm.score(train_set_x, train_set_y.values.ravel()))

train_set_x_np = train_set_x.to_numpy()
train_set_x_np = create_windows(train_set_x_np, window_shape=look_back, end_id=-look_ahead)

train_set_y_np = train_set_y.to_numpy()
train_set_y_np = create_windows(train_set_y_np, window_shape=look_back, end_id=-look_ahead)

rnn.fit(train_set_x_np, train_set_y_np, validation_data=(train_set_x_np, train_set_y_np))
RNN_predict = rnn.predict(train_set_x_np)

brnn.fit(train_set_x_np, train_set_y_np, validation_data=(train_set_x_np, train_set_y_np))
BRNN_predict = brnn.predict(train_set_x_np)

lstm.fit(train_set_x_np, train_set_y_np, validation_data=(train_set_x_np, train_set_y_np))
LSTM_predict = lstm.predict(train_set_x_np)

gru.fit(train_set_x_np, train_set_y_np, validation_data=(train_set_x_np, train_set_y_np))
GRU_predict = gru.predict(train_set_x_np)

level2_dataset_final_train = pd.DataFrame(columns=['RF', 'ERT', 'XGB', 'LGBM', 'RNN', 'BRNN', 'LSTM', 'GRU', 'label'])

for i in range(0, train_set_y.shape[0]):
    if level2_dataset_final_train.empty:  # check if the new dataset is empty or not
        if train_set_y.iloc[0, 0] == 0:  # Check if true label is 0 or not, if it is zero we use the first column of
            # our produced prediction probability
            level2_dataset_final_train.at[0, 'RF'] = random_forest_predict[0, 0]
            level2_dataset_final_train.at[0, 'ERT'] = extreme_random_tree_predict[0, 0]
            level2_dataset_final_train.at[0, 'XGB'] = xg_boost_predict[0, 0]
            level2_dataset_final_train.at[0, 'LGBM'] = light_gbm_predict[0, 0]
            level2_dataset_final_train.at[0, 'label'] = train_set_y.iloc[0, 0]

            level2_dataset_final_train.at[0, 'RNN'] = None
            level2_dataset_final_train.at[0, 'BRNN'] = None
            level2_dataset_final_train.at[0, 'LSTM'] = None
            level2_dataset_final_train.at[0, 'GRU'] = None
        else:
            level2_dataset_final_train.at[0, 'RF'] = random_forest_predict[0, 1]
            level2_dataset_final_train.at[0, 'ERT'] = extreme_random_tree_predict[0, 1]
            level2_dataset_final_train.at[0, 'XGB'] = xg_boost_predict[0, 1]
            level2_dataset_final_train.at[0, 'LGBM'] = light_gbm_predict[0, 1]
            level2_dataset_final_train.at[0, 'label'] = train_set_y.iloc[0, 0]

            level2_dataset_final_train.at[0, 'RNN'] = None
            level2_dataset_final_train.at[0, 'BRNN'] = None
            level2_dataset_final_train.at[0, 'LSTM'] = None
            level2_dataset_final_train.at[0, 'GRU'] = None

    else:  # if the new DataFrame is not empty we should find the last index to use attaching new data

        if train_set_y.iloc[i, 0] == 0:

            level2_dataset_final_train.at[i, 'RF'] = random_forest_predict[i, 0]
            level2_dataset_final_train.at[i, 'ERT'] = extreme_random_tree_predict[i, 0]
            level2_dataset_final_train.at[i, 'XGB'] = xg_boost_predict[i, 0]
            level2_dataset_final_train.at[i, 'LGBM'] = light_gbm_predict[i, 0]
            level2_dataset_final_train.at[i, 'label'] = train_set_y.iloc[i, 0]

            level2_dataset_final_train.at[i, 'RNN'] = RNN_predict[i - 1, 0]  # changed from 1- to 0
            level2_dataset_final_train.at[i, 'BRNN'] = BRNN_predict[i - 1, 0]
            level2_dataset_final_train.at[i, 'LSTM'] = LSTM_predict[i - 1, 0]
            level2_dataset_final_train.at[i, 'GRU'] = GRU_predict[i - 1, 0]
        else:

            level2_dataset_final_train.at[i, 'RF'] = random_forest_predict[i, 1]
            level2_dataset_final_train.at[i, 'ERT'] = extreme_random_tree_predict[i, 1]
            level2_dataset_final_train.at[i, 'XGB'] = xg_boost_predict[i, 1]
            level2_dataset_final_train.at[i, 'LGBM'] = light_gbm_predict[i, 1]
            level2_dataset_final_train.at[i, 'label'] = train_set_y.iloc[i, 0]

            level2_dataset_final_train.at[i, 'RNN'] = RNN_predict[i - 1, 0]
            level2_dataset_final_train.at[i, 'BRNN'] = BRNN_predict[i - 1, 0]
            level2_dataset_final_train.at[i, 'LSTM'] = LSTM_predict[i - 1, 0]
            level2_dataset_final_train.at[i, 'GRU'] = GRU_predict[i - 1, 0]

level2_dataset_final_train = level2_dataset_final_train.dropna()
level2_dataset_final_train.reset_index(inplace=True, drop=True)

level2_x_final_train = level2_dataset_final_train.loc[:, level2_dataset_final_train.columns != 'label']
level2_y_final_train = level2_dataset_final_train.loc[:, level2_dataset_final_train.columns == 'label']
level2_y_final_train = level2_y_final_train.astype('int')

logistic_model.fit(level2_x_final_train, level2_y_final_train.values.ravel())

# final Test Phase
random_forest_predict_test = random_forest.predict_proba(test_set_x)

extreme_random_tree_predict_test = extreme_random_tree.predict_proba(test_set_x)

xg_boost_predict_test = xg_boost.predict_proba(test_set_x)

light_gbm_predict_test = light_gbm.predict_proba(test_set_x)

test_set_x_np = test_set_x.to_numpy()
test_set_x_np = create_windows(test_set_x_np, window_shape=look_back, end_id=-look_ahead)

test_set_y_np = test_set_y.to_numpy()
test_set_y_np = create_windows(test_set_y_np, window_shape=look_back, end_id=-look_ahead)

RNN_predict_test = rnn.predict(test_set_x_np)

BRNN_predict_test = brnn.predict(test_set_x_np)

LSTM_predict_test = lstm.predict(test_set_x_np)

GRU_predict_test = gru.predict(test_set_x_np)

level2_dataset_final_test = pd.DataFrame(columns=['RF', 'ERT', 'XGB', 'LGBM', 'RNN', 'BRNN', 'LSTM', 'GRU', 'label'])

for i in range(0, test_set_y.shape[0]):
    if level2_dataset_final_test.empty:  # check if the new dataset is empty or not
        if test_set_y.iloc[0, 0] == 0:  # Check if true label is 0 or not, if it is zero we use the first column of
            # our produced prediction probability
            level2_dataset_final_test.at[0, 'RF'] = random_forest_predict_test[0, 0]
            level2_dataset_final_test.at[0, 'ERT'] = extreme_random_tree_predict_test[0, 0]
            level2_dataset_final_test.at[0, 'XGB'] = xg_boost_predict_test[0, 0]
            level2_dataset_final_test.at[0, 'LGBM'] = light_gbm_predict_test[0, 0]
            level2_dataset_final_test.at[0, 'label'] = test_set_y.iloc[0, 0]

            level2_dataset_final_test.at[0, 'RNN'] = None
            level2_dataset_final_test.at[0, 'BRNN'] = None
            level2_dataset_final_test.at[0, 'LSTM'] = None
            level2_dataset_final_test.at[0, 'GRU'] = None
        else:
            level2_dataset_final_test.at[0, 'RF'] = random_forest_predict_test[0, 1]
            level2_dataset_final_test.at[0, 'ERT'] = extreme_random_tree_predict_test[0, 1]
            level2_dataset_final_test.at[0, 'XGB'] = xg_boost_predict_test[0, 1]
            level2_dataset_final_test.at[0, 'LGBM'] = light_gbm_predict_test[0, 1]
            level2_dataset_final_test.at[0, 'label'] = test_set_y.iloc[0, 0]

            level2_dataset_final_test.at[0, 'RNN'] = None
            level2_dataset_final_test.at[0, 'BRNN'] = None
            level2_dataset_final_test.at[0, 'LSTM'] = None
            level2_dataset_final_test.at[0, 'GRU'] = None

    else:  # if the new DataFrame is not empty we should find the last index to use attaching new data

        if test_set_y.iloc[i, 0] == 0:

            level2_dataset_final_test.at[i, 'RF'] = random_forest_predict_test[i, 0]
            level2_dataset_final_test.at[i, 'ERT'] = extreme_random_tree_predict_test[i, 0]
            level2_dataset_final_test.at[i, 'XGB'] = xg_boost_predict_test[i, 0]
            level2_dataset_final_test.at[i, 'LGBM'] = light_gbm_predict_test[i, 0]
            level2_dataset_final_test.at[i, 'label'] = test_set_y.iloc[i, 0]

            level2_dataset_final_test.at[i, 'RNN'] = RNN_predict_test[i - 1, 0]
            level2_dataset_final_test.at[i, 'BRNN'] = BRNN_predict_test[i - 1, 0]
            level2_dataset_final_test.at[i, 'LSTM'] = LSTM_predict_test[i - 1, 0]
            level2_dataset_final_test.at[i, 'GRU'] = GRU_predict_test[i - 1, 0]
        else:

            level2_dataset_final_test.at[i, 'RF'] = random_forest_predict_test[i, 1]
            level2_dataset_final_test.at[i, 'ERT'] = extreme_random_tree_predict_test[i, 1]
            level2_dataset_final_test.at[i, 'XGB'] = xg_boost_predict_test[i, 1]
            level2_dataset_final_test.at[i, 'LGBM'] = light_gbm_predict_test[i, 1]
            level2_dataset_final_test.at[i, 'label'] = test_set_y.iloc[i, 0]

            level2_dataset_final_test.at[i, 'RNN'] = RNN_predict_test[i - 1, 0]
            level2_dataset_final_test.at[i, 'BRNN'] = BRNN_predict_test[i - 1, 0]
            level2_dataset_final_test.at[i, 'LSTM'] = LSTM_predict_test[i - 1, 0]
            level2_dataset_final_test.at[i, 'GRU'] = GRU_predict_test[i - 1, 0]

level2_dataset_final_test_star = pd.concat([level2_dataset_final_test, test_set_x], axis=1)
level2_dataset_final_test_star = level2_dataset_final_test_star.dropna()
level2_dataset_final_test_star.reset_index(inplace=True, drop=True)

level2_dataset_final_test = level2_dataset_final_test.dropna()
level2_dataset_final_test.reset_index(inplace=True, drop=True)

#  plotting the ERT vs LSTM
index_list = level2_dataset_final_test.index.values.tolist()
ERT_list = level2_dataset_final_test["ERT"].values.tolist()
LSTM_list = level2_dataset_final_test["LSTM"].values.tolist()

f = plt.figure()
f.set_figwidth(40)
f.set_figheight(5)
plt.plot(index_list, ERT_list, label="ERT")
plt.plot(index_list, LSTM_list, label="LSTM")

plt.legend()
plt.show()

level2_x_final_test = level2_dataset_final_test.loc[:, level2_dataset_final_test.columns != 'label']
level2_y_final_test = level2_dataset_final_test.loc[:, level2_dataset_final_test.columns == 'label']
level2_y_final_test = level2_y_final_test.astype('int')

prd = logistic_model.predict(level2_x_final_test)

precision = precision_score(level2_y_final_test, prd, average='binary')
# print('Precision: %.3f' % precision)

recall = recall_score(level2_y_final_test, prd, average='binary')
# print('Recall: %.3f' % recall)

accuracy = accuracy_score(level2_y_final_test, prd)
print('Accuracy: %.3f' % accuracy)

score = f1_score(level2_y_final_test, prd, average='binary')
print('F-Measure: %.3f' % score)

fpr, tpr, _ = roc_curve(level2_y_final_test, prd, pos_label=1)
auc1 = auc(fpr, tpr)
print('AUC : %.3f' % auc1)

# Lasso* , Combining the Original features with the prediction probabilities from the first level classifiers
level2_dataset_final_train_star = pd.concat([level2_dataset_final_train, train_set_x], axis=1)
level2_dataset_final_train_star = level2_dataset_final_train_star.dropna()
level2_dataset_final_train_star.reset_index(inplace=True, drop=True)

logistic_model_star = LogisticRegression(penalty="l1", C=0.1, solver="liblinear")  #

level2_x_final_train_star = level2_dataset_final_train_star.loc[:, level2_dataset_final_train_star.columns != 'label']
level2_y_final_train_star = level2_dataset_final_train_star.loc[:, level2_dataset_final_train_star.columns == 'label']
level2_y_final_train_star = level2_y_final_train_star.astype('int')

logistic_model_star.fit(level2_x_final_train_star, level2_y_final_train_star.values.ravel())

level2_x_final_test_star = level2_dataset_final_test_star.loc[:, level2_dataset_final_test_star.columns != 'label']
level2_y_final_test_star = level2_dataset_final_test_star.loc[:, level2_dataset_final_test_star.columns == 'label']
level2_y_final_test_star = level2_y_final_test_star.astype('int')

prd_star = logistic_model_star.predict(level2_x_final_test_star)

accuracy = accuracy_score(level2_y_final_test, prd_star)
print('Accuracy: %.3f' % accuracy)

score = f1_score(level2_y_final_test, prd_star, average='binary')
print('F-Measure: %.3f' % score)

fpr, tpr, _ = roc_curve(level2_y_final_test, prd_star, pos_label=1)
auc1 = auc(fpr, tpr)
print('AUC : %.3f' % auc1)
