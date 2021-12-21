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
test = pd.read_csv('test_data_matrix.csv')

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
    # No interactions
    #for x in x_vars:
        #cop_dat[x + '_FemInt'] = cop_dat[x]*cop_dat['sexf']
        #cop_dat[x + '_WhiteInt'] = cop_dat[x]*cop_dat['racewhite']
    cop_dat['FemWhite'] = cop_dat['sexf']*cop_dat['racewhite']
    cop_dat = cop_dat.astype(int)
    return cop_dat

orig_miss = prep_data(orig)
orig_test = prep_data(test)

# Creating minority and male dummy variable for metrics
orig_miss['MinorityDummy'] = 1*(orig['racewhite'] == 0)
orig_miss['MaleDummy'] = 1*(orig['sexf'] == 0)
orig_test['MinorityDummy'] = 1*(orig_test['racewhite'] == 0)
orig_test['MaleDummy'] = 1*(orig_test['sexf'] == 0)

train = orig_miss
test = orig_test
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
final_models['LogitNoReg'] = LogisticRegression(penalty='none', solver='newton-cg')
final_models['LogitL1'] = LogisticRegression(penalty='l1', solver='liblinear')
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
    bs, fp, fin_metric = fairness_funcs.fairness_metric(train[v],train[y_var], train[min_var])
    # What happens if we just adjust probs to always be below 0.5
    bsc, fpc, fin_metricc = fairness_funcs.fairness_metric(train[v].clip(0,0.4999),train[y_var],train[min_var])
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
    metm, metf = fairness_funcs.fairness_metric_sex(train,v,y_var,min_var,sex_var)
    # What happens if we just adjust probs to always be below 0.5
    print('LOOKING AT METRICS WHEN CLIPPING PROBABILITIES')
    metmc, metfc = fairness_funcs.fairness_metric_sex(train,v,y_var,min_var,sex_var,clip=True)
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

# Creating the final prediction model, not sure to use L1 or logit no penalty
# Will go with L1

picked_mod = 'LogitL1' #'LogitNoReg'

fin_dat = test[['id',picked_mod]].copy()
fin_dat.rename(columns={picked_mod: 'Probability', 'id':'ID'}, 
               inplace=True)
fin_dat['Probability'] = fin_dat['Probability'].round(4).clip(0,0.4999)
fin_dat.to_csv('L1Predictions_forTest.csv', index=False)
#fin_dat.to_xls

