#######################################
# Trying out discrete time long
#######################################

###############################################
# Libraries and set up

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime

import sys
sys.path.append(r'D:\Dropbox\Dropbox\nij_forecasting')
import fairness_funcs #my functions
import andy_helpers #mostly plotting functions
from pytorch_mods import pytorchLogit #my pytorch models

import os
os.chdir(r'D:\Dropbox\Dropbox\nij_forecasting')

np.random.seed(10)
###############################################

train = pd.read_csv('train_data_matrix2.csv')
test = pd.read_csv('test_data_matrix3.csv')

train.fillna(0, inplace=True)
test.fillna(0,inplace=True)

# Only want to train on those not failed in round 1 or round 2
train_nofail = train[(train['y2'] + train['y1']) == 0].copy().reset_index(drop=True)
train_nofail['MinorityDummy'] = 1*(train_nofail['racewhite'] == 0)

rand_sel = np.random.uniform(size=train_nofail.shape[0]) > 0.85


train_eval = train_nofail[~rand_sel].copy()
train_hold = train_nofail[rand_sel].copy()


y_var = 'y3'
x_vars = list(train)[5:]

final_models = {}
final_models['XGB_nest1000_depth10'] = XGBClassifier(n_estimators=1000, max_depth=10)
final_models['XGB_nest1000_depth10'] = XGBClassifier(n_estimators=1000, max_depth=3)
final_models['XGB_nest100_depth10'] = XGBClassifier(n_estimators=100, max_depth=10)
final_models['XGB_nest50_depth10'] = XGBClassifier(n_estimators=100, max_depth=5)
final_models['XGB_nest50_depth10_Brier'] = XGBClassifier(n_estimators=100, max_depth=5, objective=fairness_funcs.brier)
final_models['RF_10md_ms50'] = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=50)
final_models['RF_4md_ms100'] = RandomForestClassifier(n_estimators=1000, max_depth=4, min_samples_split=100)
final_models['RF_4md_ms100'] = RandomForestClassifier(n_estimators=1000, max_depth=4, min_samples_split=100)
final_models['RF_3md_ms10'] = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_split=10)
final_models['RF_5md_ms10'] = RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_split=10)
final_models['LogitNoReg'] = LogisticRegression(penalty='none', solver='newton-cg')
final_models['LogitElastic'] = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')
final_models['LogitL2'] = LogisticRegression(penalty='l2')
final_models['LogitL1'] = LogisticRegression(penalty='l1', solver='liblinear')

# Iterating over each model and fitting
for nm, mod in final_models.items():
    print(f'Fitting Model {nm} start {datetime.now()}')
    mod.fit(train_eval[x_vars], train_eval[y_var])

# Adding predicted probabilities back into original datasets
pred_prob_cols = list(final_models.keys()) #variable names
for nm, mod in final_models.items():
    # Predicted probs for in sample
    train_eval[nm] = mod.predict_proba(train_eval[x_vars])[:,1]
    # Predicted probs out of sample
    train_hold[nm] =  mod.predict_proba(train_hold[x_vars])[:,1]
###############################################



# Out of sample
min_var = 'MinorityDummy'
metrics = {} #need to append these to make a dataframe
for v in pred_prob_cols:
    print(f'\nMETRICS FOR {v}')
    min_exp, maj_exp = fairness_funcs.fpr_groups_pred(train_hold[v], train_hold[v], train_hold[min_var])
    bs, fp, fin_metric = fairness_funcs.fairness_metric(train_hold[v],train_hold[y_var], train_hold[min_var])
    # What happens if we just adjust probs to always be below 0.5
    bsc, fpc, fin_metricc = fairness_funcs.fairness_metric(train_hold[v].clip(0,0.4999),train_hold[y_var],train_hold[min_var])
    # This is better than original lol!
    metrics[v] = [min_exp[0], min_exp[1], min_exp[2], maj_exp[0], maj_exp[1], maj_exp[2], 
                  bs, fp, fin_metric, bsc, fpc, fin_metricc]

col_names = ['ExpMinCnt','ExpTotMinor','ExpMinRate',
             'ExpMajCnt','ExpTotMajor','ExpMajRate',
             'BS','FP_Diff','FinMetric',
             'BS_Clip','FP_Clip','FinMetricClip']

full_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=col_names)
print(full_metrics)

# Lets look at calibration and ROC/AUC
andy_helpers.cal_data_wide(probs=pred_prob_cols, true=y_var, data=train_hold,
                           bins=12, plot=True, wrap_col=3, sns_height=4, save_plot=False)

andy_helpers.auc_plot(data=train_hold, y_true=y_var, y_scores=pred_prob_cols, 
                      leg_size= 'x-small')

# Yeah based on both of these will still go with logit l1
# with the clipped probabilities

fin_mod = LogisticRegression(penalty='l1', solver='liblinear')
fin_mod.fit(train[x_vars], train[y_var])
test['Probability'] = fin_mod.predict_proba(test[x_vars])[:,1]

fin_dat = test[['id','Probability']].copy()
fin_dat.rename(columns={'id':'ID'}, 
               inplace=True)
fin_dat['Probability'] = fin_dat['Probability'].round(4).clip(0,0.4999)
fin_dat.to_csv('L1Predictions_forTest_Round3.csv', index=False)
