from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from lightgbm import LGBMClassifier
import torch
import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # to Ignore the deprication Warnings


def absolute_maximum_scale(series):
    """A function for  Normalize the data"""
    return series / series.abs().max()


data = pd.read_excel("/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine "
                     "Learning/Project/codes/dow_full.xlsx")
X = data.loc[:, data.columns != 'label']  # dividing the data to feature matrix
Y = data.loc[:, data.columns == 'label']  # labels of data
X = X.drop(columns=['Unnamed: 0', 'Date', 'Open', 'High', 'Low', 'Close'], axis=1)  # eliminating Useless Features
for col in X.columns:
    X[col] = absolute_maximum_scale(X[col])

train_set_x = X[0:2700]  # first 2700 records separated for training purposes
train_set_x.reset_index(inplace=True, drop=True)
train_set_y = Y[0:2700]
train_set_y.reset_index(inplace=True, drop=True)

test_set_x = X[2700:]  # rest of the data for final test phase
test_set_x.reset_index(inplace=True, drop=True)
test_set_y = Y[2700:]
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

# Recurrent neural network (RNN)
rnn = torch.nn.RNN(input_size=1, hidden_size=128, num_layers=3, batch_first=False)

# Bidirectional RNN (BRNN)
brnn = torch.nn.RNN(input_size=1, hidden_size=128, num_layers=3, batch_first=False, bidirectional=True, dropout=0.3)

# Long short-term memory (LSTM)
lstm = torch.nn.LSTM(input_size=1, hidden_size=128, num_layers=3, batch_first=False, dropout=0.3)

# Gated recurrent unit (GRU)
gru = torch.nn.GRU(input_size=1, hidden_size=128, num_layers=3, batch_first=False)

level2_dataset = pd.DataFrame(columns=['RF', 'ERT', 'XGB', 'LGBM', 'RNN', 'BRNN', 'LSTM', 'GRU', 'label'])  # new DataFrae to for data for next level classifier (Meta Classifier)

for train_index, validation_index in kf8.split(train_set_x, train_set_y, ):  # using train and validation set in CV
    x_train = train_set_x.iloc[train_index]  # training set , provided by CV
    y_train = train_set_y.iloc[train_index]

    x_valid = train_set_x.iloc[validation_index]  # validation set provided by CV
    y_valid = train_set_y.loc[validation_index]

# fitting the base level classifiers with training set
    random_forest.fit(x_train, y_train.values.ravel())
    extreme_random_tree.fit(x_train, y_train.values.ravel())
    xg_boost.fit(x_train, y_train.values.ravel())
    light_gbm.fit(x_train, y_train.values.ravel())

    # rnn.fit(x_train, y_train.values.ravel())
    # brnn.fit(x_train, y_train.values.ravel())
    # lstm.fit(x_train, y_train.values.ravel())
    # gru.fit(x_train, y_train.values.ravel())

# extracting the prediction probability on validation set
    random_forest_predict = random_forest.predict_proba(x_valid)
    print("RF : ", random_forest.score(x_valid, y_valid.values.ravel()))
    extreme_random_tree_predict = extreme_random_tree.predict_proba(x_valid)
    print("ERT : ", extreme_random_tree.score(x_valid, y_valid.values.ravel()))
    xg_boost_predict = xg_boost.predict_proba(x_valid)
    print("XGBoost : ", xg_boost.score(x_valid, y_valid.values.ravel()))
    light_gbm_predict = light_gbm.predict_proba(x_valid)
    print("LightGBM : ", light_gbm.score(x_valid, y_valid.values.ravel()))

    # RNN_predict = rnn.predict_proba(x_valid)
    # print("RNN : ", rnn.score(x_valid, y_valid.values.ravel()))

    # BRNN_predict = brnn.predict_proba(x_valid)
    # print("BRNN : ", brnn.score(x_valid, y_valid.values.ravel()))

    # LSTM_predict = lstm.predict_proba(x_valid)
    # print("LSTM : ", lstm.score(x_valid, y_valid.values.ravel()))

    # GRU_predict = gru.predict_proba(x_valid)
    # print("GRU : ", gru.score(x_valid, y_valid.values.ravel()))

# using the produced prediction probabilities to form news data set for Meta classifier
    for i in range(0, y_valid.shape[0]):
        if level2_dataset.empty:  # check if the new dataset is empty or not
            if y_valid.iloc[i, 0] == 0:  # Check if true label is 0 or not, if it is zero we use the first column of
                # our produced prediction probability
                level2_dataset.at[i, 'RF'] = random_forest_predict[i, 0]
                level2_dataset.at[i, 'ERT'] = extreme_random_tree_predict[i, 0]
                level2_dataset.at[i, 'XGB'] = xg_boost_predict[i, 0]
                level2_dataset.at[i, 'LGBM'] = light_gbm_predict[i, 0]
                level2_dataset.at[i, 'label'] = y_valid.iloc[i, 0]
                # level2_dataset.at[i, 'RNN'] = RNN_predict[i, 0]
                # level2_dataset.at[i, 'BRNN'] = BRNN_predict[i, 0]
                # level2_dataset.at[i, 'LSTM'] = LSTM_predict[i, 0]
                # level2_dataset.at[i, 'GRU'] = GRU_predict[i, 0]
            else:
                level2_dataset.at[i, 'RF'] = random_forest_predict[i, 1]
                level2_dataset.at[i, 'ERT'] = extreme_random_tree_predict[i, 1]
                level2_dataset.at[i, 'XGB'] = xg_boost_predict[i, 1]
                level2_dataset.at[i, 'LGBM'] = light_gbm_predict[i, 1]
                level2_dataset.at[i, 'label'] = y_valid.iloc[i, 0]
                # level2_dataset.at[i, 'RNN'] = RNN_predict[i, 1]
                # level2_dataset.at[i, 'BRNN'] = BRNN_predict[i, 1]
                # level2_dataset.at[i, 'LSTM'] = LSTM_predict[i, 1]
                # level2_dataset.at[i, 'GRU'] = GRU_predict[i, 1]

        else:  # if the new DataFrame is not empty we should find the last index to use attaching new data
            level2_dataset.reset_index(inplace=True, drop=True)
            level2_dataset_ind = level2_dataset.index[-1] + 1
            if y_valid.iloc[i, 0] == 0:
                level2_dataset.at[level2_dataset_ind + i, 'RF'] = random_forest_predict[i, 0]
                level2_dataset.at[level2_dataset_ind + i, 'ERT'] = extreme_random_tree_predict[i, 0]
                level2_dataset.at[level2_dataset_ind + i, 'XGB'] = xg_boost_predict[i, 0]
                level2_dataset.at[level2_dataset_ind + i, 'LGBM'] = light_gbm_predict[i, 0]
                level2_dataset.at[level2_dataset_ind + i, 'label'] = y_valid.iloc[i, 0]
                # level2_dataset.at[level2_dataset_ind + i, 'RNN'] = RNN_predict[i, 0]
                # level2_dataset.at[level2_dataset_ind + i, 'BRNN'] = BRNN_predict[i, 0]
                # level2_dataset.at[level2_dataset_ind + i, 'LSTM'] = LSTM_predict[i, 0]
                # level2_dataset.at[level2_dataset_ind + i, 'GRU'] = GRU_predict[i, 0]
            else:
                level2_dataset.at[level2_dataset_ind + i, 'RF'] = random_forest_predict[i, 1]
                level2_dataset.at[level2_dataset_ind + i, 'ERT'] = extreme_random_tree_predict[i, 1]
                level2_dataset.at[level2_dataset_ind + i, 'XGB'] = xg_boost_predict[i, 1]
                level2_dataset.at[level2_dataset_ind + i, 'LGBM'] = light_gbm_predict[i, 1]
                level2_dataset.at[level2_dataset_ind + i, 'label'] = y_valid.iloc[i, 0]
                # level2_dataset.at[level2_dataset_ind + i, 'RNN'] = RNN_predict[i, 1]
                # level2_dataset.at[level2_dataset_ind + i, 'BRNN'] = BRNN_predict[i, 1]
                # level2_dataset.at[level2_dataset_ind + i, 'LSTM'] = LSTM_predict[i, 1]
                # level2_dataset.at[level2_dataset_ind + i, 'GRU'] = GRU_predict[i, 1]
    level2_dataset.reset_index(inplace=True, drop=True)  # reindexing the new DataFrame

level2_x = level2_dataset.loc[:, level2_dataset.columns != 'label']
level2_y = level2_dataset.loc[:, level2_dataset.columns == 'label']
level2_y = level2_y.astype('int')

logistic_model = LogisticRegression()
logistic_model.fit(level2_x, level2_y.values.ravel())
print("score: ", logistic_model.score(level2_x, level2_y.values.ravel()))

"""pd.DataFrame(xg_boost.feature_importances_.reshape(1, -1), columns=X.columns)
y_pred = xg_boost.predict(X)
mean_squared_error(Y.values.ravel(), y_pred)
pred = light_gbm.predict(X)
accuracy = light_gbm.score(X, Y.values.ravel())
print(accuracy)
"""
