import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme()

df = pd.read_csv('result/ablation2.csv')

df['sparsity'] = (100-100*(df['nn_row']/df['mm_row'])).astype(int)

df = df[((df.error==0) & (df.bm<=df.m))].sort_values(by="speedup", ascending=False).groupby(by=["algo","mm_row","m","n","k","bm","block_sz"]).first().reset_index()


pruners = ['32','64','128']

df['sparsity'] = df['sparsity'].astype({'sparsity':'string'})
df['bm'] = df['bm'].astype(int)
df['algo'] = df['algo'].astype(int)

print(df)


df_32 = df[df['algo']==4].set_index(['sparsity','bm'])
df_32 = df_32.rename({'speedup':'32_bit'}, axis=1)
df_32 = df_32['32_bit']

df_128 = df[df['algo']==2].set_index(['sparsity','bm'])
df_128 = df_128.rename({'speedup':'128_bit'}, axis=1)
df_128 = df_128['128_bit']

df = pd.concat([df_32, df_128], axis=1)

print(df)

fig = plt.figure(figsize=(14,8), dpi=200)
ax = fig.add_subplot(111)


def plot_function(x, ax):
    ax = graph[x]
    ax.set_xlabel(x, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.vlines(0.5,-0.4,3, color='k')
    return df.xs(x).plot(kind='bar', ax=ax, legend=False)

n_subplots = len(df.index.levels[0])
fig, axes = plt.subplots(nrows=1, ncols=n_subplots, sharey=True, figsize=(14, 8),gridspec_kw={'hspace': 0.0, 'wspace': 0.05})  # width, height

graph = dict(zip(df.index.levels[0], axes))
plots = list(map(lambda x: plot_function(x, graph[x]), graph))
ax.tick_params(axis='both', which='both', length=0)
fig.subplots_adjust(wspace=0)

fig.text(0.5, 0.04, 'Sparsity [%] (V:N:M)', ha='center', fontsize=20)
fig.text(0.01, 0.55, 'Speedup w.r.t. cuBLAS', va='center', rotation='vertical', fontsize=20)
fig.subplots_adjust(left=0.125, bottom=0.25, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

plt.legend(fontsize=16)
plt.savefig('result/ablation2.pdf', bbox_inches = 'tight')
#plt.show()