import pandas as pd
from os import walk
import numpy as np

# !/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as figure
import os
import math
import numpy as np
from shapely.geometry import LineString
from os.path import exists
import seaborn as sns
sns.set_theme()


df = pd.read_csv("result/ablation1.csv")
df['sparsity'] = 100-100*(df['nn_row']/df['mm_col'])
df.loc[df.algo==4, "sparsity"] = 50.0

tmp = df.sort_values(by="speedup", ascending=False).groupby(by=["algo","mm_col","m","n","k"]).first().reset_index()

tmp['gflops'] = tmp['m']/1e9*tmp['n']*tmp['k']*2
tmp['dense_gflops'] = tmp['gflops']/(tmp['gemm_time'])
tmp['sp_gflops'] = tmp['gflops']/(tmp['spmm_time'])


fig, axes = plt.subplots(1,4, figsize=(17,6),gridspec_kw={'hspace': 2.5, 'wspace': 0.15})

pos=0

############
for m in [10,20,40,100]:
    ax = axes[pos]

    tmp2  = tmp[tmp.mm_col==m]
    tmp2=tmp2.sort_values(by=['k'])

    tmp_openSparseLt = tmp2[tmp2.algo==2]
    tmp_cuSparseLt_wo = tmp2[tmp2.algo==7]
    x = tmp_openSparseLt['k'].tolist()

    ax.bar(np.arange(len(x)), tmp_openSparseLt['speedup'].tolist(), label="w/ column-loc", width=0.4)
    ax.bar(np.arange(len(x))-0.4, tmp_cuSparseLt_wo['speedup'].tolist(), label="w/o column-loc", width=0.4)
    ax.set_xticks(range(len(x)), x, rotation=35, ha="right")
    #ax2.plot(range(len(x)), np.ones(len(x)), color="k")
    ax.set_xlabel("K"+"\n"+str(int(100-100*(2/m))) + "% [" +"128:2:"+str(m)+"]")
    pos+=1

fig.text(0.007, 0.5, 'SpeedUp w.r.t. cuBLAS', va='center', rotation='vertical', fontsize=15)
fig.text(0.5, 0.009, 'Sparsity [%] (V:N:M)', ha='center', fontsize=15)
axes[2].legend(loc='upper left', bbox_to_anchor=(.0004, 1.2),
          ncol=2, fancybox=True, shadow=True, fontsize=12)
#plt.legend(prop={'size': 15})
fig.subplots_adjust(left=0.04, bottom=0.25, right=0.99, top=0.85, wspace=0.2, hspace=0.2)
#plt.show()
plt.savefig('result/ablation1.pdf', bbox_inches = 'tight')