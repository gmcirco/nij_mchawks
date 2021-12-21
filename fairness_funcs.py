####################################
# Fairness functions
#
#
# Gio Circo & Andy Wheeler
####################################

import numpy as np
import pandas as pd
import pulp

# Brier loss, p should be clipped to 0/1 before input if using that as loss function
def brier_score(p,a):
    return ((p - a)**2).mean()

def fpr_groups_pred(prob_true, prob_adjust, minority, thresh=0.5):
    # Calculating the total over the threshold for each group
    over = 1*(prob_adjust >= thresh)         # using adjusted probabilities
    min_tot = (over*minority).sum()          # total minority over threshold
    maj_tot = (over*(minority == 0)).sum()   # total majority over threshold
    # Expected false positives for each group using true probabilities
    min_fp = (over*minority*(1 - prob_true)).sum()
    maj_fp = (over*(minority == 0)*(1 - prob_true)).sum()
    # Expected false positive rate for each subgroup
    min_fpr = min_fp/min_tot.clip(1)
    maj_fpr = maj_fp/maj_tot.clip(1)
    # Nice print function
    print(" ")
    f1 = f'''Predicted differences between groups'''
    f2 = f'''white fpr {maj_fpr:.2f} ( {maj_fp:.1f} / {maj_tot})'''
    f3 = f'''black fpr {min_fpr:.2f} ( {min_fp:.1f} / {min_tot})'''
    print(f1)
    print(f2)
    print(f3)
    print(" ")
    # Returning tuples for each
    return (min_fp, min_tot, min_fpr), (maj_fp, maj_tot, maj_fpr)

def fpr_groups_sex_pred(data,prob_true, prob_adjust,minority,male,thresh=0.5):
    # Separating out each subgroup
    dat_copy = data[[prob_true,prob_adjust,minority,male]].copy()
    male_sg = data[dat_copy[male] == 1]
    fem_sg = data[dat_copy[male] == 0]
    print('MALE SUBGROUP SCORES PREDICTED')
    minm, majm = fpr_groups_pred(male_sg[prob_true], male_sg[prob_adjust], male_sg[minority], thresh=thresh)
    print('FEMALE SUBGROUP SCORES PREDICTED')
    minf, majf = fpr_groups_pred(fem_sg[prob_true], fem_sg[prob_adjust], fem_sg[minority], thresh=thresh)
    return minm, majm, minf, majf

def fpr_groups_act(prob_adjust,actual,minority,thresh=0.5):
    # Calculating the total over the threshold for each group
    over = 1*(prob_adjust >= thresh)         # using adjusted probabilities
    min_tot = (over*minority).sum()          # total minority over threshold
    maj_tot = (over*(minority == 0)).sum()   # total majority over threshold
    # Expected false positives for each group using actual outcomes
    min_fp = (over*minority*(actual == 0)).sum()
    maj_fp = (over*(minority == 0)*(actual == 0)).sum()
    # Expected false positive rate for each subgroup
    min_fpr = min_fp/min_tot.clip(1)
    maj_fpr = maj_fp/maj_tot.clip(1)
    # Nice print function
    print(" ")
    f1 = f'''Observed differences between groups'''
    f2 = f'''white fpr {maj_fpr:.2f} ( {maj_fp} / {maj_tot})'''
    f3 = f'''black fpr {min_fpr:.2f} ( {min_fp} / {min_tot})'''
    print(f1)
    print(f2)
    print(f3)
    print(" ")
    # Returning tuples for each
    return (min_fp, min_tot, min_fpr), (maj_fp, maj_tot, maj_fpr)

def fpr_groups_sex_act(data,prob_adjust,actual,minority,male,thresh=0.5):
    # Separating out each subgroup
    dat_copy = data[[actual,prob_adjust,minority,male]].copy()
    male_sg = data[dat_copy[male] == 1]
    fem_sg = data[dat_copy[male] == 0]
    print('MALE SUBGROUP SCORES ACTUAL')
    minm, majm = fpr_groups_act(male_sg[prob_adjust], male_sg[actual], male_sg[minority], thresh=thresh)
    print('FEMALE SUBGROUP SCORES ACTUAL')
    minf, majf = fpr_groups_act(fem_sg[prob_adjust], fem_sg[actual], fem_sg[minority], thresh=thresh)
    return minm, majm, minf, majf

# Calculating fairness for two groups at threshold 0.5
def fairness_metric(prob,actual,minority,thresh=0.5):
    # Calculating the Brier score for the whole sample
    BS = brier_score(prob,actual)
    # Calculating the false positive rate for each racial group
    min_stats, maj_stats = fpr_groups_act(prob,actual,minority,thresh)
    # Now the combined loss function
    FP = 1 - np.abs( min_stats[2] - maj_stats[2] )
    acc = (1 - BS)*(FP)
    print(" ")
    print(f'Brier Score {BS:.3f}, FPR Difference {1 - FP:.3f}, overall metric {acc:.3f}')
    return BS, FP, acc

# Fairness metric broken down by sex subgroups
def fairness_metric_sex(data,prob,actual,minority,male,thresh=0.5,clip=False):
    # Separating out each subgroup
    dat_copy = data[[prob,actual,minority,male]].copy()
    if clip:
        dat_copy[prob] = dat_copy[prob].clip(0,thresh - 0.0001)
    male_sg = dat_copy[dat_copy[male] == 1]
    fem_sg = dat_copy[dat_copy[male] == 0]
    # Now doing the score for each subgroup
    print('FAIRNESS METRIC SCORE FOR MALES')
    bsm, fpm, accm = fairness_metric(male_sg[prob],male_sg[actual],male_sg[minority],thresh=0.5)
    print('FAIRNESS METRIC SCORE FOR FEMALES')
    bsf, fpf, accf = fairness_metric(fem_sg[prob],fem_sg[actual],fem_sg[minority],thresh=0.5)
    return (bsm, fpm, accm), (bsf, fpf, accf)

# Brier score loss function for XGBoost
# Via https://stackoverflow.com/a/60925043/604456
def brier(y_true, y_pred):
    labels = y_true
    preds = 1.0 / (1.0 + np.exp(-y_pred))
    grad = 2*(preds-labels)*preds*(1-preds)
    hess = 2*(2*(labels+1)*preds-labels-3*preds*preds)*preds*(1-preds)
    return grad, hess

def evalerror(preds, dtrain):
    preds = 1.0 / (1.0 + np.exp(-preds))
    labels = dtrain.get_label()
    errors = (labels - preds)**2
    return 'brier-error', np.mean(errors)


# linear program to flip a minimal number of scores
# to meet false positive rate constraint

# Grid search over FPR constraints
# to see which one maximizes the fairness metric


