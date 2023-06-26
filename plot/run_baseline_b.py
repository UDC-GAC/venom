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
sns.set_theme(style="ticks")

df = pd.read_csv("result/baseline_b.csv")
df['sparsity'] = 100-100*(df['nn_row']/df['mm_col'])
df.loc[df.algo==3, "sparsity"] = 50.0

tmp2 = df.sort_values(by="speedup", ascending=False).groupby(by=["algo","mm_col","m","n","k"]).first().reset_index()

tmp2['gflops'] = 2*tmp2['m']*tmp2['n']*tmp2['k']*1e-9
tmp2['dense_gflops'] = tmp2['gflops']/(tmp2['gemm_time'])
tmp2['sp_gflops'] = tmp2['gflops']/(tmp2['spmm_time'])

print(tmp2[tmp2.algo==3]['spmm_time']/tmp2[tmp2.algo==2]['spmm_time'])

print(tmp2)
############
fig, ax = plt.subplots(figsize=(12,8))
ax2 = ax.twinx()

tmp2=tmp2.sort_values(by=['gflops'])

tmp2['Problem size'] = tmp2['m'].astype('str')+"/"+tmp2['k'].astype('str')+"/"+tmp2['n'].astype('str')

tmp_openSparseLt = tmp2[tmp2.algo==2]
tmp_cuSparseLt = tmp2[tmp2.algo==3]
x = tmp_openSparseLt['k'].tolist()

print(tmp_openSparseLt)
print(tmp_cuSparseLt)

print(len(tmp_openSparseLt))
print(len(tmp_cuSparseLt))

ax.bar(np.arange(len(x)), tmp_openSparseLt['sp_gflops'].tolist(), 0.3, label="Spatha")
ax.bar(np.arange(len(x))-0.3, tmp_openSparseLt['dense_gflops'].tolist(), 0.3, label="cuBLAS")
ax.bar(np.arange(len(x))+0.3, tmp_cuSparseLt['sp_gflops'].tolist(), 0.3, label="cuSparseLt")

import matplotlib.patheffects as mpe
pe1 = [mpe.Stroke(linewidth=3.5, foreground='black'),
       mpe.Stroke(foreground='white',alpha=1),
       mpe.Normal()]

ax2.plot(range(len(x)), tmp_openSparseLt['speedup'].tolist(), color='r', marker="o", label="Spatha", path_effects=pe1)
ax2.plot(range(len(x)), tmp_cuSparseLt['speedup'].tolist(), "r--", marker="^", label="cuSparseLt", path_effects=pe1)
#ax2.plot(range(len(x)), np.ones(len(x)), color="k")

ax.set_xticks(range(len(x)), x, size='small', rotation=35, ha="right")
plt.ylabel('SpeedUp', color='k')
ax.set_ylabel('TFLOPS/s', fontsize=14.5)
ax.set_xlabel('K', fontsize=14.5)

ax.tick_params(axis='x', labelsize=15.5)
ax.tick_params(axis='y', labelsize=15.5)
ax2.tick_params(axis='y', labelsize=15.5)

ax2.set_ylim([1, 2])
plt.ylabel('SpeedUp w.r.t. cuBLAS', color='r', fontsize=14.5, weight='bold')
plt.xlabel('Layers MxK', fontsize=14.5, weight='bold')
#plt.title("[BERT] AVG SpeedUp: "+str(round(tmp_openSparseLt.sp.mean(), 4)))
plt.title("GEMM-M=1024, GEMM-N=4096 (BERT-large)", fontsize=14.5, y=-1.05)
ax.legend()
#ax.legend(loc="upper left", fontsize=14)
#ax2.legend(loc="upper right", fontsize=14)
ax.legend(loc='upper left', bbox_to_anchor=(0.0004, 1.25),
          ncol=2, fancybox=True, shadow=True, fontsize=15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.01, 1.25),
          ncol=1, fancybox=True, shadow=True, fontsize=15)
fig.subplots_adjust(left=0.125, bottom=0.22, right=0.85, top=0.8, wspace=0.2, hspace=0.2)
plt.savefig('result/baseline_b.pdf', bbox_inches = 'tight')
#plt.show()