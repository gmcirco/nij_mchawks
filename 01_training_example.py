#######################################
# First attempt at training and seeing
# if my fairness functions work as 
# expected
#
# Andy Wheeler
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

###############################################
# Data Prep
# Get the training data
orig = pd.read_csv('train_data_matrix .csv')

# Some missing, just imputing to -1 for now
# This is OK for tree based methods
#orig.fillna(-1, inplace=True)

# Creating dummy variables for missing
#mis = orig.isna().sum(axis=0)
#dum_vars = list(mis[mis > 0].index)

# Creating function to apply to out of sample data
def prep_data(data):
    cop_dat = data.copy()
    dum_vars = ['gang', 'risk_score', 'risk_levelspecialized', 
                'risk_levelhigh', 'prison_offenseother', 
                'prison_offenseproperty', 
                'prison_offenseviolent_non_sex', 
                'prison_offenseviolent_sex']
    for d in dum_vars:
        cop_dat[d + "_md"] = 1*data[d].isna()
    cop_dat.fillna(0, inplace=True)
    # creating interaction variables between sex and race and other variables
    ext_vars = set(['id','y1','y2','y3','yany','racewhite','sexf'])
    x_vars = list(set(list(data)) - ext_vars)
    for x in x_vars:
        cop_dat[x + '_FemInt'] = cop_dat[x]*cop_dat['sexf']
        cop_dat[x + '_WhiteInt'] = cop_dat[x]*cop_dat['racewhite']
        cop_dat['FemWhite'] = cop_dat['sexf']*cop_dat['racewhite']
    cop_dat = cop_dat.astype(int)
    return cop_dat

orig_miss = prep_data(orig)

# Creating minority and male dummy variable for metrics
orig_miss['MinorityDummy'] = 1*(orig['racewhite'] == 0)
orig_miss['MaleDummy'] = 1*(orig['sexf'] == 0)

# To make sure my functions work as expected
# doing my own train/test split
nt = 10000 #around 18k observations
train = orig_miss[orig_miss.index < nt].copy()
test = orig_miss[orig_miss.index >= nt].copy()
###############################################

###############################################
# Estimating Model and Predicted Probs on test

# Estimating xgboost model
# Not worrying about Brier score for now
# but could, see https://stackoverflow.com/q/52595782/604456

x_vars = list( set(list(orig)) - set(['id','y1','y2','y3','yany','MinorityDummy','MaleDummy']) )
y_var = 'y1'
min_var = 'MinorityDummy'
sex_var = 'MaleDummy'

# Simple tests XGBoost is worse for Brier score/calibration
# But, smaller number of over 0.5 for RandomForests (a good thing)
# Makes the fairness metric more unstable

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
final_models['logitrelu_logloss'] = pytorchLogit()
final_models['logitrelu_brier'] = pytorchLogit(loss='brier')
final_models['logittan_logloss'] = pytorchLogit(activate='tanh')
final_models['logittan_brier'] = pytorchLogit(loss='brier', activate='tanh')
final_models['logittan_brier_clamp'] = pytorchLogit(loss='brier', activate='tanh', final='clamp')
final_models['logittan_brier_fair'] = pytorchLogit(loss='brier_fair', activate='tanh')
final_models['logittan_brier_fair_clamps'] = pytorchLogit(loss='brier_fair', activate='tanh', final='clamp')
final_models['logittan_brier_fair_relu'] = pytorchLogit(loss='brier_fair', activate='relu')
final_models['logittan_brier_fair_relu_clamp'] = pytorchLogit(loss='brier_fair', activate='relu', final='clamp')

# Iterating over each model and fitting
for nm, mod in final_models.items():
    print(f'Fitting Model {nm} start {datetime.now()}')
    mod.fit(train[x_vars], train[y_var])

# Adding predicted probabilities back into original datasets
pred_prob_cols = list(final_models.keys()) #variable names
for nm, mod in final_models.items():
    # Predicted probs for in sample
    train[nm] = mod.predict_proba(train[x_vars])[:,1]
    # Predicted probs out of sample
    test[nm] =  mod.predict_proba(test[x_vars])[:,1]
###############################################

###############################################
# Seeing the Brier Score & Fairness metric for original predictions

# Out of sample
metrics = {} #need to append these to make a dataframe
for v in pred_prob_cols:
    print(f'\nMETRICS FOR {v}')
    min_exp, maj_exp = fairness_funcs.fpr_groups_pred(test[v], test[v], test[min_var])
    bs, fp, fin_metric = fairness_funcs.fairness_metric(test[v],test[y_var], test[min_var])
    # What happens if we just adjust probs to always be below 0.5
    bsc, fpc, fin_metricc = fairness_funcs.fairness_metric(test[v].clip(0,0.4999),test[y_var],test[min_var])
    # This is better than original lol!
    metrics[v] = [min_exp[0], min_exp[1], min_exp[2], maj_exp[0], maj_exp[1], maj_exp[2], 
                  bs, fp, fin_metric, bsc, fpc, fin_metricc]

col_names = ['ExpMinCnt','ExpTotMinor','ExpMinRate',
             'ExpMajCnt','ExpTotMajor','ExpMajRate',
             'BS','FP_Diff','FinMetric',
             'BS_Clip','FP_Clip','FinMetricClip']

full_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=col_names)
print(full_metrics)

# Out of sample metrics broken down by gender
metrics_m = {} #need to append these to make a dataframe
metrics_f = {}
for v in pred_prob_cols:
    print(f'\nMETRICS FOR {v}')
    min_exp_m, maj_exp_m, min_exp_f, maj_exp_f = fairness_funcs.fpr_groups_sex_pred(test, v, v, min_var,sex_var)
    metm, metf = fairness_funcs.fairness_metric_sex(test,v,y_var,min_var,sex_var)
    # What happens if we just adjust probs to always be below 0.5
    print('LOOKING AT METRICS WHEN CLIPPING PROBABILITIES')
    metmc, metfc = fairness_funcs.fairness_metric_sex(test,v,y_var,min_var,sex_var,clip=True)
    metrics_m[v] = [min_exp_m[0], min_exp_m[1], min_exp_m[2], 
                    maj_exp_m[0], maj_exp_m[1], maj_exp_m[2], 
                    metm[0], metm[1], metm[2], 
                    metmc[0], metmc[1], metmc[2], 'Male']
    metrics_f[v] = [min_exp_f[0], min_exp_f[1], min_exp_f[2], 
                    maj_exp_f[0], maj_exp_f[1], maj_exp_f[2], 
                    metf[0], metf[1], metf[2], 
                    metfc[0], metfc[1], metfc[2], 'Female']

col_names.append('Sex')

male_metrics = pd.DataFrame.from_dict(metrics_m, orient='index', columns=col_names)
fem_metrics = pd.DataFrame.from_dict(metrics_f, orient='index', columns=col_names)

male_metrics.to_csv('MaleResults_Combo.csv')
fem_metrics.to_csv('FemaleResults_Combo.csv')

######################################################
# Initial Results comparing XGBoost and random forest
# For my test dataset
#             BS   FP_Diff    Overall
#XGB        0.259    0.01      0.739  # this suggests default XGBoost is overfitting with all these variables
#XGB Clip   0.236    0         0.764  # clip means I just clipped the predictions to <0.5
#RF         0.194    0.14      0.693  # just a few predictions are barely over 0.5 for RF, so clipping makes
#RF Clip    0.194    0         0.806  # very little difference to Brier score
#Logit      0.188    0.019     0.797  # Logit with no regularization does better
#Logit Clip 0.189    0         0.811  # than l1/l2/elastic-net (Logit better calibrated)
######################################################

# Lets look at calibration and ROC/AUC
andy_helpers.cal_data_wide(probs=pred_prob_cols, true=y_var, data=test,
                           bins=30, plot=True, wrap_col=3, sns_height=4, save_plot=False)

# By racial breakdown
andy_helpers.cal_data_wide_group(probs=pred_prob_cols, true=y_var, group='racewhite', 
                    data=test, bins=30, plot=True, wrap_col=5, 
                    sns_height=3, font_title=12, save_plot=False)

# By sex breakdown

andy_helpers.auc_plot(data=test, y_true=y_var, y_scores=pred_prob_cols, 
                      leg_size= 'x-small')
andy_helpers.auc_plot_long(data=test, y_true=y_var, y_score='LogitNoReg',
                           group='racewhite')

# Example using wide and by group
#test['grp'] = np.mod(test['id'],5)
#andy_helpers.auc_plot_wide_group(data=test, y_true=y_var, y_scores=pred_prob_cols,
                    group='grp', ncols=4, size=4.5, leg_size='xx-small')

###############################################

###############################################
# My functions to adjust the predicted probability for some cases
# ?????????????????

###############################################
