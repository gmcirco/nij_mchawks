## Replication materials for Circo and Wheeler (2021) NIJ forecasting competition

The round 1 and round 3 results were fit using a Python script, while the round 2 results were fit using an R script. This
repo contains the Python and R scripts used to generate the results, while the processed data used to fit the models are saved seperately
as dropbox links (see below). In addition, the original NIJ report can be downloaded [here](https://www.crimrxiv.com/pub/bc7mptfb/release/1). 
Links to the NIJ contest page and the original data sources can be found [here](https://nij.ojp.gov/funding/recidivism-forecasting-challenge) and 
[here](https://data.ojp.usdoj.gov/stories/s/daxx-hznc).

## Replication Data

Data to replicate the results in the paper can be downloaded from the following folder:
  - [NIJ Replication Data](https://www.dropbox.com/sh/6ko4wjhmvn2g0bx/AAB0nvQ1j1P13oPcq0txqvOna?dl=0)
  
This folder contains the processed data used to fit the models. The file `train_data_matrix.csv` is the original training dataset. 
For forecasting in rounds 2 and 3, individuals who recidivated were removed, based on their absence in one of the three test datasets.
`train_data_matrix2.csv` is specifically pre-processed for use in the R replication script which will generate the results shown in the
NIJ contest paper. `full_data_matrix.csv` is the final, full dataset that NIJ released following the end of the competition. This can be used
to compare predictions against the true values. There are also a number of other scripts that were used to fit intermediate models in both R
and Python. These are included to help illustrate our analysis process. If you are interested in the results for our winning model, the 
`xgboost_final_eval.R` file contains a more consise collection of scripts to produce the analysis and figures.

## Round 1-3 Replication Code

To replicate the round 1 results you will need the following files:
  - `03_predictround1.py`
  - `andy_helpers.py`
  - `fairness_funcs.py`
  - `pytorch_mods.py`
  - `train_data_matrix.csv`
  - `test_data_matrix.csv`
  
To replicate the round 2 results and NIJ paper results you will need the following files:
  - `xgboost_final_eval.R`
  - `full_data_matrix.csv`
  - `train_data_matrix2.csv`

To replicate the round 3 results you will need the following files:
  - `04_predictround3.py`
  - `andy_helpers.py`
  - `fairness_funcs.py`
  - `pytorch_mods.py`
  - `train_data_matrix2.csv`
  - `test_data_matrix3.csv`
