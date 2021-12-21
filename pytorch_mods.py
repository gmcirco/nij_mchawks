#######################################
# Pytorch logit modelels
#
# Andy Wheeler
#######################################

###############################################
# Libraries and set up

import numpy as np
import pandas as pd
import torch
torch.manual_seed(10)

# Setting the device globally
try:
    device = torch.device("cuda:0")
    print(f'Torch device default to cuda:0')
except:
    device = torch.device("cpu")
    print(f'No cuda, torch device default to cpu')

def set_device(x):
    global device
    device = torch.device(x)

###############################################

###############################################
# Creating pytorch models to mimic sklearn fit

def tclamp(input):
    return torch.clamp(input,min=0,max=1)

# brier score loss function with fairness constraint
def brier_fair(pred,obs,minority,thresh=0.5):
    bs = ((pred - obs)**2).mean()
    over = 1*(pred > thresh)
    majority = 1*(minority == 0)
    fp = 1*over*(obs == 0)
    min_tot = (over*minority).sum().clamp(1)
    maj_tot = (over*majority).sum().clamp(1)
    min_fp = (fp*minority).sum()
    maj_fp = (fp*majority).sum()
    min_rate = min_fp/min_tot
    maj_rate = maj_fp/maj_tot
    diff_rate = torch.abs(min_rate - maj_rate)
    fin_score = (1 - bs)*(1 - diff_rate)
    return -fin_score

class logit_pytorch(torch.nn.Module):
    def __init__(self, nvars, activate='relu', bias=True,
                 final='sigmoid', device=device):
        """
        Construct parameters for the coefficients 
        activate - either string ('relu' or 'tanh', 
                   or pass in your own torch function
        bias - whether to include bias (intercept) in model
        final - use either 'sigmoid' to squash to probs, or 'clamp'
                or pass in your own torch function
        device - torch device to construct the tensors
                 default cuda:0 if available
        """
        super(logit_pytorch, self).__init__()
        self.coef = torch.nn.Parameter(torch.rand((nvars,1),
                    device=device)/10)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,
                    device=device))
        else:
            self.bias = torch.zeros(1, device=device)
        if activate == 'relu':
            self.trans = torch.nn.ReLU()
        elif activate == 'tanh':
            self.trans = torch.nn.Tanh()
        else:
            self.trans = activate
        if final == 'sigmoid':
            self.final = torch.nn.Sigmoid()
        elif final == 'clamp':
            self.final = tclamp
        else: 
            self.final = final
    def forward(self, x):
        """
        predicted probability
        """
        output = self.bias + torch.mm(x, self.trans(self.coef))
        return self.final(output)

class pytorchLogit():
    def __init__(self, loss='logit', iters=25000, 
                 activate='relu', bias=True, 
                 final='sigmoid', device=device,
                 minority='racewhite', threshold=0.5,
                 printn=1000):
        """
        loss - either string 'logit' or 'brier' or own pytorch function
        iters - number of iterations to fit (default 25000)
        activate - either string ('relu' or 'tanh', 
                   or pass in your own torch function
        bias - whether to include bias (intercept) in model
        final - use either 'sigmoid' to squash to probs, or 'clamp'
                or pass in your own torch function. Should not use clamp
                with default logit loss
        opt - ?optimizer?
        device - torch device to construct the tensors
                 default cuda:0 if available
        printn - how often to check the fit (default 1000 iters)
        """
        super(pytorchLogit, self).__init__()
        if loss == 'logit':
            self.loss = torch.nn.BCELoss()
            self.loss_name = 'logit'
        elif loss == 'brier':
            self.loss = torch.nn.MSELoss(reduction='mean')
            self.loss_name = 'brier'
        elif loss == 'brier_fair':
            self.loss = brier_fair
            self.loss_name = 'brier_fair'
        else:
            self.loss = loss
            self.loss_name = 'user defined function'
        self.threshold = threshold
        self.iters = iters
        self.mod = None
        self.activate = activate
        self.bias = bias
        self.final = final
        self.device = device
        self.printn = 1000
        self.minority = minority
    def fit(self, X, y):
        x_ten = torch.tensor(X.to_numpy(), dtype=torch.float,
                             device=device)
        y_ten = torch.tensor(pd.DataFrame(y).to_numpy(), dtype=torch.float,
                             device=device)
        # Only needed for fair loss function
        if self.loss_name == 'brier_fair':
            min_ten = torch.tensor(X[[self.minority]].to_numpy(), dtype=torch.int,
                                   device=device)
        # If mod is not already created, create a new one, else update prior
        if self.mod is None:
            loc_mod = logit_pytorch(nvars=X.shape[1], activate=self.activate, 
                                    bias=self.bias, final=self.final, 
                                    device=self.device)
            # Appending the model object (maybe should dump to numpy)
            self.mod = loc_mod
        else:
            loc_mod = self.mod
        opt = torch.optim.Adam(loc_mod.parameters(), lr=1e-4)
        crit = self.loss
        if self.loss_name == 'brier_fair':
            # Do a burn in period for the brier loss
            brier_loss = torch.nn.MSELoss(reduction='mean')
            for t in range(2000):
                opt.zero_grad()
                y_pred = loc_mod(x_ten)
                bl = brier_loss(y_pred,y_ten)
                bl.backward()
                opt.step()
            # Now go to the fairness loss function
            for t in range(self.iters):
                opt.zero_grad()
                y_pred = loc_mod(x_ten)
                loss = crit(y_pred,y_ten,min_ten,self.threshold)
                if t % self.printn == 99:
                    print(f'{t}: {loss.item():.5f}')
                loss.backward()
                opt.step()
        else:
            for t in range(self.iters):
                opt.zero_grad()
                y_pred = loc_mod(x_ten)
                loss = crit(y_pred,y_ten)
                if t % self.printn == 99:
                    print(f'{t}: {loss.item():.5f}')
                loss.backward()
                opt.step()
    def predict_proba(self, X):
        x_ten = torch.tensor(X.to_numpy(), dtype=torch.float,
                             device=device)
        res = self.mod(x_ten)
        pp = res.cpu().detach().numpy()
        return np.concatenate((1-pp,pp), axis=1)