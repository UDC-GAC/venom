#
# Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_theme()

df = pd.read_csv("result/inference.csv")
print(df)
width = 0.5

figure, ax = plt.subplots(1, 3, figsize=(11,8))
bottom = np.zeros(4)

tmp=df[(df.algo==0) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
print(tmp)
ax[0].set_title("BERT-large, bs=32")
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax[0], legend=False)
ax_df.set_xticklabels(["dense", "64:2:8", "64:2:16", "64:2:32"])
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

tmp=df[(df.algo==1) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[1].set_title("GPT2-large, bs=8")
#ax[1].set_xticks(["dense", "64:2:8", "64:2:16", "64:2:32"])
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax[1], legend=False)
ax_df.set_xticklabels(["dense", "64:2:8", "64:2:16", "64:2:32"])
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

tmp=df[(df.algo==2) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[2].set_title("GPT3, bs=1")
#ax[2].set_xticks(["dense", "64:2:8", "64:2:16", "64:2:32"])
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax[2], legend=False)
ax_df.set_xticklabels(["dense", "64:2:8", "64:2:16", "64:2:32"])
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

#ax.set_title("Number of penguins with above average body mass")
figure.tight_layout()
ax[1].set_xlabel("Sparsity", fontsize=14)
ax[0].set_ylabel("Latency(ms)", fontsize=14)

figure.subplots_adjust(left=0.125, bottom=0.22, right=0.95, top=0.85, wspace=0.2, hspace=0.2)
#plt.show()
plt.savefig('result/inference_v64.pdf')


##############
figure, ax = plt.subplots(1, 3, figsize=(11,8))
bottom = np.zeros(4)

tmp=df[(df.algo==0) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
print(tmp)
ax[0].set_title("BERT-large, bs=32")
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax[0], legend=False)
ax_df.set_xticklabels(["dense", "128:2:8", "128:2:16", "128:2:32"])
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

tmp=df[(df.algo==1) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[1].set_title("GPT2-large, bs=8")
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax[1], legend=False)
ax_df.set_xticklabels(["dense", "128:2:8", "128:2:16", "128:2:32"])
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

tmp=df[(df.algo==2) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[2].set_title("GPT3, bs=1")
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax[2], legend=False)
ax_df.set_xticklabels(["dense", "128:2:8", "128:2:16", "128:2:32"])
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

#ax.set_title("Number of penguins with above average body mass")
figure.tight_layout()
ax[1].set_xlabel("Sparsity", fontsize=14)
ax[0].set_ylabel("Latency(ms)", fontsize=14)

figure.subplots_adjust(left=0.125, bottom=0.22, right=0.95, top=0.85, wspace=0.2, hspace=0.2)
#plt.show()
plt.savefig('result/inference_v128.pdf')