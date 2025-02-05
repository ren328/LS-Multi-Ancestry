import math
import scipy
import scipy.linalg 
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


##set parameters
omega1=0 ## can be changed
omega2=1 ##  can be changed

##white individuals
r = robjects.r
r['source']('path')
rfunction = robjects.globalenv['f1']
with localconverter(robjects.default_converter + pandas2ri.converter):
     snp1= rfunction(i)

snp1=np.array(snp1)
##white GWAS
beta1=np.loadtxt("")

p1=30000

res=[]
for j in range(p1):
  c=snp1[:,j]
  cp=np.matmul(c.T,c)
  cp1=1/cp
  res.append(cp1)
res1=np.diag(res)
cxt1=np.matmul(res1,snp1.T)
xxt2=np.matmul(cxt1.T,cxt1)
xxt3=omega1*xxt2
temp2=np.matmul(cxt1.T,beta1)   
temp3=omega1*temp2

##black individuals
r = robjects.r
r['source']('')
rfunction = robjects.globalenv['f2']
with localconverter(robjects.default_converter + pandas2ri.converter):
     snp= rfunction(i)

snp=np.array(snp)
## black GWAS
beta=np.loadtxt("")


p=30000
res=[]
for j in range(p):
  c=snp[:,j]
  cp=np.matmul(c.T,c)
  cp1=1/cp
  res.append(cp1)

res1=np.diag(res)
cxt=np.matmul(res1,snp.T)
xxt=np.matmul(cxt.T,cxt)
xxt1=omega2*xxt

temp=np.matmul(cxt.T,beta)   
temp1=omega2*temp


term1=xxt1+xxt3

a1=np.diag(term1)
lambda1=1e-6
a2=a1+lambda1
np.fill_diagonal(term1,a2)

xxtinv=np.linalg.inv(term1)

term2=temp1+temp3

yhat=np.matmul(xxtinv,term2)    




## save data
np.savetxt("",yhat)

