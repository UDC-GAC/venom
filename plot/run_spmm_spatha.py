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
import seaborn as sns
sns.set_theme(style="ticks")

from shapely.geometry import LineString
from os.path import exists

import holoviews as hv
import pandas as pd
from bokeh.models import Span
import panel as pn
pn.extension()

fig, axes = plt.subplots(2,4, figsize=(14,6), gridspec_kw={'hspace': 0.1, 'wspace': 0.0}, sharey='row', sharex='col') # Create matplotlib figure

df = pd.read_csv("result/spmm_bert_spatha.csv")

df['sparsity'] = (100-100*(df['nn_row']/df['mm_col']))

df.loc[df.algo==3, "sparsity"] = 50.0

a = df[((df.error==0) & (df.bm<=df.m)) | (df.algo==3)].sort_values(by="speedup", ascending=False).groupby(by=["algo","mm_col","m","n","k","bm","block_sz"]).first().reset_index()

print("*********************************************************")
levels=["50%", "70%", "75%", "80%", "90%", "95%", "98%"]

th = [4//2, 7//2, 8//2, 10//2, 20//2, 40//2, 100//2]


counter=0
counter_idx = [(1,0), (1,1)]
#cols2=[128,256,512]
#cols2=[2048,4096,8192]
cols2=[4096,8192]

#sns.set_style("darkgrid")
print(a)

df2 = a[(a.m==768)|(a.m==3072)]
a = a[(a.m==4096)|(a.m==1024)]
for bs in cols2:
    a_aux  = a[a.n==bs]

    counter_x, counter_y = counter_idx[counter]
    ax = axes[counter_x, counter_y]

    for e, c, lib, style in zip([0,-0.3,0.3,0.6], ["red", "green", "blue", "orange"], ["Spatha", "cuSparseLt", "Sputnik", "CLASP"], ["--",'*','--','--']):

        #c="green";lib="openSparseLt"; style="-"
        if lib=="Spatha":
            a1  = a_aux[(a_aux.algo==2) & (a_aux.bm==64)]
        elif lib=="Sputnik":
            a1  = a_aux[a_aux.algo==1]
        elif lib=="CLASP":
            a1  = a_aux[(a_aux.algo==0) & (a_aux.block_sz==4)]
        else:
            a1  = a_aux[a_aux.algo==3]
        print(a1)

        a2 = a1.groupby(by=["sparsity"])
        b = a2.mean()
        #b = a1.median()
        y1 = b['speedup']

        ax.set_yscale('log')

        if lib!="cuSparseLt":
            if e==0:
                ax.plot(levels, np.ones(len(y1)), label="cuBLAS", color="green")
            ax.plot(levels, y1, label=lib, color=c, linestyle=style)
            #ax.plot(levels, th, color="k", linestyle="--")
        else:
            print(y1)
            ax.scatter(e, y1, label=lib, color=c, marker=style, s=18)
            #pass

        if lib!="cuSparseLt":
            props = dict(boxes=c, whiskers=c, medians=c, caps="Gray")
            boxplot = a1.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, widths=.3, color=props, notch=False)
            #boxplot = sns.boxplot(data=a1, y='speedup', x='sparsity', ax=ax)
            #boxplot = sns.boxplot(x = 'sparsity', y = 'speedup', data = a, color = "green", ax=ax)
            #boxplot = a.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, color=props)
            boxplot.set(xticklabels=[])
            boxplot.tick_params(bottom=False)
            boxplot.get_figure().gca().set_title("")
            boxplot.get_figure().suptitle('')
            #[[item.set_color('k') for item in bp[key]['medians']] for key in boxplot.keys()]

        title = "N=" + str(bs)
        ax.set_title('')
        ylabel = "t="
        ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        #ax.set_ylabel(ylabel, fontsize=11)

        fig.subplots_adjust(left=0.07, bottom=0.185, right=0.896, top=0.844, wspace=0.2, hspace=0.2)

        ax.set_xticks(levels)
        ax.set_xticklabels(levels, rotation=25, fontsize=10)

        lines, labels = ax.get_legend_handles_labels()


        ax.set_xlabel('')
        #ax.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    counter+=1
##########################
counter_idx = [(1,2), (1,3)]
counter=0
for bs in cols2:
    a_aux  = a[a.n==bs]

    counter_x, counter_y = counter_idx[counter]
    ax = axes[counter_x, counter_y]

    for e, c, lib, style in zip([0,-0.3,0.3,0.6], ["red", "green", "blue", "orange"], ["Spatha", "cuSparseLt", "Sputnik", "CLASP"], ["--",'*','--','--']):

        #c="green";lib="openSparseLt"; style="-"
        if lib=="Spatha":
            a1  = a_aux[(a_aux.algo==2)&(a_aux.bm==128)]
        elif lib=="Sputnik":
            a1  = a_aux[a_aux.algo==1]
        elif lib=="CLASP":
            a1  = a_aux[(a_aux.algo==0)&(a_aux.block_sz==8)]
        else:
            a1  = a_aux[a_aux.algo==3]
        print(a1)

        a2 = a1.groupby(by=["sparsity"])
        b = a2.mean()
        #b = a1.median()
        y1 = b['speedup']

        ax.set_yscale('log')

        if lib!="cuSparseLt":
            if e==0:
                ax.plot(levels, np.ones(len(y1)), label="cuBLAS", color="green")
            ax.plot(levels, y1, label=lib, color=c, linestyle=style)
            #ax.plot(levels, th, color="k", linestyle="--")
        else:
            print(y1)
            ax.scatter(e, y1, label=lib, color=c, marker=style, s=18)
            #pass

        if lib!="cuSparseLt":
            props = dict(boxes=c, whiskers=c, medians=c, caps="Gray")
            boxplot = a1.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, widths=.3, color=props, notch=False)
            #boxplot = sns.boxplot(data=a1, y='speedup', x='sparsity', ax=ax)
            #boxplot = sns.boxplot(x = 'sparsity', y = 'speedup', data = a, color = "green", ax=ax)
            #boxplot = a.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, color=props)
            boxplot.set(xticklabels=[])
            boxplot.tick_params(bottom=False)
            boxplot.get_figure().gca().set_title("")
            boxplot.get_figure().suptitle('')
            #[[item.set_color('k') for item in bp[key]['medians']] for key in boxplot.keys()]

        title = "N=" + str(bs)
        ax.set_title('')
        ylabel = "t="
        ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        #ax.set_ylabel(ylabel, fontsize=11)

        fig.subplots_adjust(left=0.07, bottom=0.185, right=0.896, top=0.844, wspace=0.2, hspace=0.2)

        ax.set_xticks(levels)
        ax.set_xticklabels(levels, rotation=25, fontsize=10)

        lines, labels = ax.get_legend_handles_labels()


        ax.set_xlabel('')
        #ax.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    counter+=1
##########################
counter_idx = [(0,0), (0,1)]
counter=0
for bs in cols2:
    a_aux  = df2[df2.n==bs]

    counter_x, counter_y = counter_idx[counter]
    ax = axes[counter_x, counter_y]

    for e, c, lib, style in zip([0,-0.3,0.3,0.6], ["red", "green", "blue", "orange"], ["Spatha", "cuSparseLt", "Sputnik", "CLASP"], ["--",'*','--','--']):

        #c="green";lib="openSparseLt"; style="-"
        if lib=="Spatha":
            a1  = a_aux[(a_aux.algo==2)&(a_aux.bm==64)]
        elif lib=="Sputnik":
            a1  = a_aux[a_aux.algo==1]
        elif lib=="CLASP":
            a1  = a_aux[(a_aux.algo==0)&(a_aux.block_sz==4)]
        else:
            a1  = a_aux[a_aux.algo==3]
        print(a1)

        a2 = a1.groupby(by=["sparsity"])
        b = a2.mean()
        #b = a1.median()
        y1 = b['speedup']

        ax.set_yscale('log')

        if lib!="cuSparseLt":
            if e==0:
                ax.plot(levels, np.ones(len(y1)), label="cuBLAS", color="green")
            ax.plot(levels, y1, label=lib, color=c, linestyle=style)
            #ax.plot(levels, th, color="k", linestyle="--")
        else:
            print(y1)
            ax.scatter(e, y1, label=lib, color=c, marker=style, s=18)
            #pass

        if lib!="cuSparseLt":
            props = dict(boxes=c, whiskers=c, medians=c, caps="Gray")
            boxplot = a1.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, widths=.3, color=props, notch=False)
            #boxplot = sns.boxplot(data=a1, y='speedup', x='sparsity', ax=ax)
            #boxplot = sns.boxplot(x = 'sparsity', y = 'speedup', data = a, color = "green", ax=ax)
            #boxplot = a.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, color=props)
            boxplot.set(xticklabels=[])
            boxplot.tick_params(bottom=False)
            boxplot.get_figure().gca().set_title("")
            boxplot.get_figure().suptitle('')
            #[[item.set_color('k') for item in bp[key]['medians']] for key in boxplot.keys()]

        title = "N=" + str(bs)
        ax.set_title('')
        ylabel = "t="
        ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        #ax.set_ylabel(ylabel, fontsize=11)

        fig.subplots_adjust(left=0.07, bottom=0.185, right=0.896, top=0.844, wspace=0.2, hspace=0.2)

        ax.set_xticks(levels)
        ax.set_xticklabels(levels, rotation=25, fontsize=10)

        lines, labels = ax.get_legend_handles_labels()


        ax.set_xlabel('')
        #ax.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    counter+=1
##########################
counter_idx = [(0,2), (0,3)]
counter=0
for bs in cols2:
    a_aux  = df2[df2.n==bs]

    counter_x, counter_y = counter_idx[counter]
    ax = axes[counter_x, counter_y]

    for e, c, lib, style in zip([0,-0.3,0.3,0.6], ["red", "green", "blue", "orange"], ["Spatha", "cuSparseLt", "Sputnik", "CLASP"], ["--",'*','--','--']):

        #c="green";lib="openSparseLt"; style="-"
        if lib=="Spatha":
            a1  = a_aux[(a_aux.algo==2)&(a_aux.bm==128)]
        elif lib=="Sputnik":
            a1  = a_aux[a_aux.algo==1]
        elif lib=="CLASP":
            a1  = a_aux[(a_aux.algo==0)&(a_aux.block_sz==8)]
        else:
            a1  = a_aux[a_aux.algo==3]
        print(a1)

        a2 = a1.groupby(by=["sparsity"])
        b = a2.mean()
        #b = a1.median()
        y1 = b['speedup']

        ax.set_yscale('log')

        if lib!="cuSparseLt":
            if e==0:
                ax.plot(levels, np.ones(len(y1)), label="cuBLAS", color="green")
            ax.plot(levels, y1, label=lib, color=c, linestyle=style)
            #ax.plot(levels, th, color="k", linestyle="--")
        else:
            print(y1)
            ax.scatter(e, y1, label=lib, color=c, marker=style, s=18)
            #pass

        if lib!="cuSparseLt":
            props = dict(boxes=c, whiskers=c)#, medians="DarkBlue", caps="Gray")
            boxplot = a1.boxplot(column='speedup', by='sparsity', positions=[x for x in list(range(len(levels)))], ax=ax, widths=.3, color=props, notch=False)

            boxplot.set(xticklabels=[])
            boxplot.tick_params(bottom=False)
            boxplot.get_figure().gca().set_title("")
            boxplot.get_figure().suptitle('')

        title = "N=" + str(bs)
        ax.set_title('')
        ylabel = "t="
        ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        #ax.set_ylabel(ylabel, fontsize=11)

        fig.subplots_adjust(left=0.07, bottom=0.185, right=0.896, top=0.844, wspace=0.2, hspace=0.2)

        ax.set_xticks(levels)
        ax.set_xticklabels(levels, rotation=25, fontsize=10)

        lines, labels = ax.get_legend_handles_labels()


        ax.set_xlabel('')
        #ax.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    counter+=1
##########################

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12)

x=0
label_v=[(64,4), (128,8)]
for axes, col in zip(axes, cols2):
    if x==0:
        yticks=[0.15,0.3,0.5,1,2,4,6,10,15,25]
    else:
        yticks=[0.15,0.3,0.5,1,2,4,6,10,15,28]
    y=0
    for ax in axes:
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if x==0:
            v1,v2=label_v[y//2]
            ax.set_title("batch size="+str(int(cols2[y%2]/512)) + "\n" + str(v1)+":N:M" + " , vw_"+str(v2), fontsize=15)
            y+=1
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=12, rotation=0); #
    x+=1

""" axes.set_xlabel('Sparsity', fontsize=11)
axes.set_ylabel('SpeedUp', fontsize=11) """
ax.get_yaxis().set_tick_params(which='minor', size=0)
ax.get_yaxis().set_tick_params(which='minor', width=0)

#ax.set_yticks([1,2,3,4,6,8,10])
#ax.set_yticks([1,4,10,15,20,25,30,50])
fig.text(0.5, 0.009, 'Sparsity [%]', ha='center', fontsize=14)
fig.text(0.007, 0.7, 'SpeedUp (log scale)\n        BERT-base', va='center', rotation='vertical', fontsize=14)
fig.text(0.007, 0.3, 'SpeedUp (log scale)\n       BERT-large', va='center', rotation='vertical', fontsize=14)

fig.tight_layout()
plt.savefig('result/spmm_spatha.pdf', bbox_inches = 'tight')
#plt.show()