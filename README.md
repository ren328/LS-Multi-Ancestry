This is the code for Large-scale genotype-based trait imputation with multi-ancestry GWAS data.

## For LS-Imputation-Combined method, see Combine.py.

1.We need the SNP data(for both  where we want to do the imputation to be
ready in RData format, and changed the following 'path' into your path to load the data;
r['source']('path')


2. We need GWAS summary data for two different ancestry, for example, the White ancestry GWAS should be saved as txt file, only marginal effect
sizes are needed for each SNP, the order should the same as the order in the column of SNPs loaded in the first step. The two GWAS could be loaded by adding the path in "" in the below code.

white GWAS:
beta1=np.loadtxt("")

black GWAS:
beta=np.loadtxt("")

3. The final result could be saved by changing the path in  "" below.
np.savetxt("",yhat)

4. Notice that the omega1,omega2, p1,p2 could be changed as needed.

## For LS-Imputation-Transfer method for continuous trait, see Transfer_AUC.py.
1.We need the SNP data(for both  where we want to do the imputation to be
ready in RData format, and changed the following 'path' into your path to load the data;
r['source']('path')

2. We need the observed trait values to calculate the AUC score, which could be loaded  by changing the path in "" below
df = pd.read_csv("", header=None, na_values='NA')


3. We need GWAS summary data to be saved as txt file, only marginal effect
sizes are needed for each SNP, the order should the same as the order in the column of SNPs loaded in the first step. The GWAS could be loaded by adding the path in "" in the below code.

beta=np.loadtxt("")

4. The imputed trait values with single Ancestry(e.g. White Ancestry) could be loaded by changing the path below
yhat1=np.loadtxt("")

5. The imputed trait values at the last epoch, auc change with epoch, and the imputed trait values with best AUC could be saved by adding
path to the following three:

np.savetxt("",final_Y)

np.savetxt("",auc_history)

np.savetxt("", best_Y)


6. The parameters could be changed as needed, including p, learning_rate, num_epochs  and tolerance

## For LS-Imputation-Transfer method for binary trait, see Transfer_cor.py.
1.We need the SNP data(for both  where we want to do the imputation to be
ready in RData format, and changed the following 'path' into your path to load the data;
r['source']('path')

2. We need the observed trait values to calculate the AUC score, which could be loaded  by changing the path in "" below
df = pd.read_csv("", header=None, na_values='NA')


3. We need GWAS summary data to be saved as txt file, only marginal effect
sizes are needed for each SNP, the order should the same as the order in the column of SNPs loaded in the first step. The GWAS could be loaded by adding the path in "" in the below code.

beta=np.loadtxt("")

4. The imputed trait values with single Ancestry(e.g. White Ancestry) could be loaded by changing the path below
yhat1=np.loadtxt("")

5. The imputed trait values at the last epoch, correlation change with epoch and the imputed trait values with best AUC could be saved by adding
path to the following three:

np.savetxt("",final_Y)

np.savetxt("",cor_history)

np.savetxt("", best_Y)


6. The parameters could be changed as needed, including p, learning_rate, num_epochs  and tolerance











