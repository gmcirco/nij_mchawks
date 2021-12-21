#######################################
# Training light gbm with categorical
# also rf with label encoding
# and svm with radial basis
#
# Andy Wheeler
#######################################

###############################################
# Libraries and set up

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from datetime import datetime

import sys
sys.path.append(r'D:\Dropbox\Dropbox\nij_forecasting')
import fairness_funcs #my functions
import andy_helpers #mostly plotting functions
from pytorch_mods import pytorchLogit

import os
os.chdir(r'D:\Dropbox\Dropbox\nij_forecasting')

np.random.seed(10)
###############################################

###############################################
# Data Prep
# Get the training data
orig = pd.read_csv('train_data.csv')

# Numeric Impute
num_imp = ['gang','risk_score']

# Ordinal Encode (just keep puma as is)
ord_enc = {}
ord_enc['sex'] = {'m':1, 'f':0}
ord_enc['race'] = {'white':0, 'black':1}
ord_enc['age'] = {'18-22':6,'23-27':5,'28-32':4,
                  '33-37':3,'38-42':2,'43-47':1,
                  '48_older':0}
ord_enc['risk_level'] = {'standard':0,'high':1,
                         'specialized':2,'NA':-1}
ord_enc['edu_level'] = {'less_hs':0,'hs':1,'some_coll':2}
ord_enc['prison_offense'] = {'NA':-1,'drug':0,'other':1,
                             'property':2,'violent_sex':3,
                             'violent_non_sex':4}
ord_enc['prison_years'] = {'0-1years':0,'1-2years':1,
                           '2-3years':2,'greater_3years':3}

# _more clip 
more_clip = ['depend','p_arrest_felony','p_arrest_misd',
             'p_arrest_violent','p_arrest_property',
             'p_arrest_drug','p_arrest_prob',
             'p_convict_felony','p_convict_misd',
             'p_convict_property','p_convict_drug']

# Function to prep data as I want, label encode
# And missing imputation
def prep_data(data):
    cop_dat = data.copy()
    # Numeric impute
    for n in num_imp:
        cop_dat[n] = data[n].fillna(-1).astype(int)
    # Ordinal Recodes
    for o in ord_enc.keys():
        cop_dat[o] = data[o].fillna('NA').replace(ord_enc[o])
    # _more clip
    for m in more_clip:
        cop_dat[m] = data[m].str.replace('_more','').astype(int)
    return cop_dat.astype(int)

orig = prep_data(orig)

orig.describe().T

# To make sure my functions work as expected
# doing my own train/test split
nt = 10000 #around 18k observations
train = orig[orig.index < nt].copy()
test = orig[orig.index >= nt].copy()
###############################################

###############################################
# Estimating Model and Predicted Probs on test

x_vars = list( set(list(orig)) - set(['y1','y2','y3','yany']) ) #for kicks see how ID improves 'id'
y_var = 'y1'
maj_var = 'race'
cat_vars = ['puma','sex','race','risk_level','prison_offense']

# Simple tests XGBoost is worse for Brier score/calibration
# But, smaller number of over 0.5 for RandomForests (a good thing)
# Makes the fairness metric more unstable

final_models = {}
final_models['logitrelu_logloss'] = pytorchLogit()
final_models['logitrelu_brier'] = pytorchLogit(loss='brier')
final_models['logittan_logloss'] = pytorchLogit(activate='tanh')
final_models['logittan_brier'] = pytorchLogit(loss='brier', activate='tanh')
final_models['logittan_brier_clamp'] = pytorchLogit(loss='brier', activate='tanh', final='clamp')
#final_models['SVC_radial'] = SVC(kernel='rbf',probability=True)
#final_models['SVC_poly'] = SVC(kernel='poly',probability=True)
#final_models['SVC_lin'] = SVC(kernel='linear',probability=True)
final_models['light_cats'] = LGBMClassifier()
final_models['light_lin'] = LGBMClassifier()
#final_models['XGB_nest1000_depth10'] = XGBClassifier(n_estimators=1000, max_depth=10)
#final_models['XGB_nest1000_depth10'] = XGBClassifier(n_estimators=1000, max_depth=3)
#final_models['XGB_nest100_depth10'] = XGBClassifier(n_estimators=100, max_depth=10)
#final_models['XGB_nest50_depth10'] = XGBClassifier(n_estimators=100, max_depth=5)
#final_models['XGB_nest50_depth10_Brier'] = XGBClassifier(n_estimators=100, max_depth=5, objective=fairness_funcs.brier)
final_models['RF_10md_ms50'] = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=50)
#final_models['RF_4md_ms100'] = RandomForestClassifier(n_estimators=1000, max_depth=4, min_samples_split=100)
#final_models['RF_4md_ms100'] = RandomForestClassifier(n_estimators=1000, max_depth=4, min_samples_split=100)
#final_models['RF_3md_ms10'] = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_split=10)
#final_models['RF_5md_ms10'] = RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_split=10)
final_models['LogitNoReg'] = LogisticRegression(penalty='none', solver='newton-cg')
#final_models['LogitElastic'] = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')
#final_models['LogitL2'] = LogisticRegression(penalty='l2')
final_models['LogitL1'] = LogisticRegression(penalty='l1', solver='liblinear')

# Iterating over each model and fitting
for nm, mod in final_models.items():
    print(f'Fitting Model {nm} start {datetime.now()}')
    if nm == 'light_cats':
        mod.fit(train[x_vars],train[y_var],feature_name=x_vars,categorical_feature=cat_vars)
    else:
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
minority = 1*(test[maj_var] == 1) #creating MINORITY dummy variable
minority_train = 1*(train[maj_var] == 0)

# Out of sample
metrics = {} #need to append these to make a dataframe
for v in pred_prob_cols:
    print(f'\nMETRICS FOR {v}')
    min_exp, maj_exp = fairness_funcs.fpr_groups_pred(test[v], test[v], minority)
    #min_obs, maj_obs = fairness_funcs.fpr_groups_act(test[v],test[y_var],minority)
    bs, fp, fin_metric = fairness_funcs.fairness_metric(test[v],test[y_var],minority)
    # What happens if we just adjust probs to always be below 0.5
    bs, fp, fin_metric = fairness_funcs.fairness_metric(test[v].clip(0,0.4999),test[y_var],minority)
    # This is better than original lol!

######################################################
# Results for SVC and lightGBM
# For my test dataset
#                         BS   FP_Diff    Overall
#SVCRadial               0.197   0.009      0.795
#SVCRadClip              0.198   0          0.802
#SVCPoly                 0.195   0.022      0.787
#SVCPolyClip             0.196   0          0.804
#LGBMcat                 0.191   0.008      0.802
#LGBMcatClip             0.191   0          0.809
#LGBMlin                 0.192   0.009      0.801
#LGBMlinClip             0.191   0          0.809
#RF10depth50split        0.189   0.023      0.793
#RF10depth50splitclip    0.189   0          0.811
#LogitNoReg              0.187   0.053      0.770
#LogitNoRegClip          0.188   0          0.812
######################################################

# Lets look at calibration and ROC/AUC
andy_helpers.cal_data_wide(probs=pred_prob_cols, true=y_var, data=test,
                           bins=30, plot=True, wrap_col=4, sns_height=4, save_plot=False)

andy_helpers.auc_plot(data=test, y_true=y_var, y_scores=pred_prob_cols, 
                      leg_size= 'x-small')

keep_mods = ['SVC_poly','light_cats','light_lin','RF_10md_ms50','LogitNoReg']

pd.plotting.scatter_matrix(test[keep_mods], alpha=0.2)

###################################################
# Trying out ensemble predictions
ens_pred = test[keep_mods].mean(axis=1)
min_exp, maj_exp = fairness_funcs.fpr_groups_pred(ens_pred, test[v], minority)
bs, fp, fin_metric = fairness_funcs.fairness_metric(ens_pred,test[y_var],minority)
# What happens if we just adjust probs to always be below 0.5
bs, fp, fin_metric = fairness_funcs.fairness_metric(ens_pred.clip(0,0.4999),test[y_var],minority)
# Not any better than just plain old logit
###################################################