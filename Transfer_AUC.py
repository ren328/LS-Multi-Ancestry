import math
import scipy
import scipy.linalg 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score 

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


##load observed data
df = pd.read_csv("", header=None, na_values='NA')

ytrue =df.to_numpy().flatten()  

# Define the loss function
def loss_function():
    prediction = torch.matmul(X_torch, Y_torch)
    loss = torch.sum((beta_torch - prediction)**2)
    return loss



##load snp data
r = robjects.r
r['source']('path')
rfunction = robjects.globalenv['f1']
with localconverter(robjects.default_converter + pandas2ri.converter):
     snp= rfunction(i)

snp=np.array(snp)

##set parameter
p=30000
##DX'
res=[]
for j in range(p):
  c=snp[:,j]
  cp=np.matmul(c.T,c)
  cp1=1/cp
  res.append(cp1)

res1=np.diag(res)

cxt=np.matmul(res1,snp.T)

X_torch = torch.tensor(cxt, dtype=torch.float32)

##load GWAS data
beta=np.loadtxt("")

# Convert data to PyTorch tensors
beta_torch = torch.tensor(beta, dtype=torch.float32)



##load imputed trait values with White Ancestry
yhat1=np.loadtxt("")
Y_torch = nn.Parameter(torch.tensor(yhat1, dtype=torch.float32))  # Y as a parameter to optimize

# Use SGD optimizer to minimize the loss
learning_rate = 0.01
num_epochs = 100000
tolerance = 1e-8  # Tolerance for stopping criterion
optimizer = optim.SGD([Y_torch], lr=learning_rate)
loss_history = []
auc_history=[]
max_auc = -float('inf')
best_Y = None 
prev_loss = float('inf')

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    optimizer.step()

    # Record loss and check stopping criterion
    loss_value = loss.item()
    loss_history.append(loss_value)
    
    # Compute AUC score
    Y_flat = Y_torch.detach().numpy().flatten()
    mask = ~np.isnan(ytrue)
    ytrue_filtered = ytrue[mask]           # Filter out rows with NA in ytrue
    Y_flat_filtered = Y_flat[mask]  
    
    auc = roc_auc_score(ytrue_filtered, Y_flat_filtered)
    auc_history.append(auc)   

    # Check if the current correlation is the highest so far
    if auc > max_auc:
        max_auc = auc
        best_Y = Y_torch.detach().numpy().copy()  # Store the best prediction result



    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value}")

    # Check stopping criterion based on loss change
    if epoch > 0 and abs(loss_value - prev_loss) < tolerance:
        print(f"Stopping criterion met. Change in loss: {abs(loss_value - prev_loss)}")
        break
    
    prev_loss = loss_value

final_Y = Y_torch.data.numpy()
##save data
np.savetxt("",final_Y)
np.savetxt("",auc_history)
np.savetxt("", best_Y)


