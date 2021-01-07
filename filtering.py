import os, sys
import pandas as pd
import numpy as np
import glob
import pickle, json
import pylab as plt
from matplotlib.pyplot import get_cmap

from scipy.sparse import csr_matrix

src_dir = os.path.join(os.pardir, 'src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics
from filter_words import nwd_H_shuffle


fname_read = sys.argv[0]
print(fname_read)
path_read = "/projects/b1038/Pulmonary/ZiyouRen/Thesis/Paul_AM_DvsH_simulation" ##change to your directory. 
data_label = fname_read.split('.')[0]
filename = os.path.join(path_read,fname_read)
df_wd = pd.read_csv(filename,index_col=0,na_values=0).dropna(how='all',axis=0).to_sparse() ## contains the data

N_s = 100 ## set to 100 or so. ## number of random realziatoin

## make csr matrix
n_wd_csr = csr_matrix(df_wd.to_coo()).astype('int')
V,D = n_wd_csr.shape

## get entropy measure
result_H = nwd_H_shuffle(n_wd_csr,N_s=N_s)

## make dataframe and store all the values
df=pd.DataFrame(index = df_wd.index )

df['F'] = result_H['F-emp']
df['I'] = result_H['H-null-mu'] - result_H['H-emp']

## get entropy and random entropy too
df['H'] = result_H['H-emp']
df['H-tilde'] =  result_H['H-null-mu']
df['H-tilde_std'] =  result_H['H-null-std']
df['N'] = np.array(n_wd_csr.sum(axis=1))[:,0] ## number of counts
df.sort_values(by='I',ascending=False)
df[(df['N']>10**3)&(df['N']<10**4)].sort_values(by='I',ascending=False)
path_save = os.path.join(os.pardir,'output')
fname_save = 'TNF_%s_Ns%s.csv'%(data_label,N_s)
filename = os.path.join(path_save,fname_save)
df.to_csv(filename)