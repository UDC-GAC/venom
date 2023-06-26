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
sns.set_theme()

plt.rcParams.update({'font.size': 22})

species = (
    "dense",
    "128:2:8",
    "128:2:16",
    "128:2:32",
)

#####################################################################
# Result table provided by Pytorch profiler for each LLM #
# (run end2end/run_inference.sh)
#####################################################################
weight_counts_bs32 = {
    "others":  np.array([64, 91.0, 100.44, 100]),
    "softmax": np.array([15.24, 15.36, 15.357, 15.35]),
    "matmul":  np.array([22.78, 22.85, 24.74, 22.76]),
    "GEMMs":   np.array([147.631, 43.19, 22.79, 14.83]),
}

weight_counts_gpt = {
    "others":  np.array([61, 73, 73, 73]),
    "softmax": np.array([7.368, 7.362, 7.364, 7.48]),
    "matmul":  np.array([11.68, 11.63, 11.64, 11.66]),
    "GEMMs":   np.array([88.37, 26.83, 14.04, 8.151]),
}

weight_counts_gpt3 = {
    "others":  np.array([33.26, 40.33, 40.259, 39.89]),
    "softmax": np.array([8.382, 8.761, 8.734, 8.74]),
    "matmul":  np.array([15.52, 15.511, 15.72, 15.725]),
    "GEMMs":   np.array([218.23, 87.136,41.986, 21.662]),
}

width = 0.5

figure, ax = plt.subplots(1, 3)

bottom = np.zeros(4)
ax[0].set_title("BERT-large, bs=32")
for boolean, weight_count in weight_counts_bs32.items():
    p = ax[0].bar(species, weight_count, width, label=boolean, bottom=bottom)
    ax[0].set_xticklabels(species, rotation=30, ha='right', fontsize=13)
    bottom += weight_count

ax[1].set_title("GPT2-large, bs=8")
bottom = np.zeros(4)
for boolean, weight_count in weight_counts_gpt.items():
    p = ax[1].bar(species, weight_count, width, label=boolean, bottom=bottom)
    ax[1].set_xticklabels(species, rotation=30, ha='right', fontsize=13)
    bottom += weight_count

ax[2].set_title("GPT3, bs=1")
bottom = np.zeros(4)
for boolean, weight_count in weight_counts_gpt3.items():
    p = ax[2].bar(species, weight_count, width, label=boolean, bottom=bottom)
    ax[2].set_xticklabels(species, rotation=30, ha='right', fontsize=13)
    bottom += weight_count
ax[0].legend(loc='upper center', bbox_to_anchor=(1.8, 1.4),
          ncol=5, fancybox=True, shadow=True)

#ax.set_title("Number of penguins with above average body mass")
figure.tight_layout()
ax[1].set_xlabel("Sparsity", fontsize=14)
ax[0].set_ylabel("Latency(ms)", fontsize=14)
plt.savefig('result/inference.pdf')