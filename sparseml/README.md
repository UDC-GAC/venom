Example script to deploy sparse training on resources: ```sparseml.sh```

# 2nd order Pruning Implementations (sparseml/pytorch/sparsification/pruning/)

- ```modifier_pruning_obs.py```: original obs implementation adapted to 8-block pruning

- ```modifier_pruning_obs_28.py```: one-shot 2:8 pruning implementation
- ```modifier_pruning_obs_216.py```: one-shot 2:16 pruning implementation

- ```modifier_pruning_obs_28_gradual.py```: gradual 2:8 pruning, m-combinatorial version of canonical vectors
- ```modifier_pruning_obs_216_gradual.py```: gradual 2:16 pruning, m-combinatorial version of canonical vectors

- ```modifier_pruning_28_pairwise_obs.py```: gradual 2:8 pruning, pair-wise version of canonical vectors
- ```modifier_pruning_216_pairwise_obs.py```: gradual 2:16 pruning, pair-wise version of canonical vectors

- ```modifier_pruning_obs_28v64.py```: one-shot 64:2:8 pruning implementation
- ```modifier_pruning_obs_28v128.py```: one-shot 128:2:8 pruning implementation
- ```modifier_pruning_obs_216v64.py```: one-shot 64:2:16 pruning implementation
- ```modifier_pruning_obs_216v128.py```: one-shot 128:2:16 pruning implementation

- ```modifier_pruning_obs_28v64_gradual.py```: gradual 64:2:8 pruning, m-combinatorial version of canonical vectors
- ```modifier_pruning_obs_28v128_gradual.py```: gradual 128:2:8 pruning, m-combinatorial version of canonical vectors
- ```modifier_pruning_obs_216v64_gradual.py```: gradual 64:2:16 pruning, m-combinatorial version of canonical vectors
- ```modifier_pruning_obs_216v128_gradual.py```: gradual 128:2:16 pruning, m-combinatorial version of canonical vectors

- ```modifier_28_pairwise_obs_v64.py```: gradual 64:2:8 pruning, pair-wise version of canonical vectors
- ```modifier_28_pairwise_obs_v128.py```: gradual 64:2:8 pruning, pair-wise version of canonical vectors
- ```modifier_216_pairwise_obs_v64.py```: gradual 64:2:16 pruning, pair-wise version of canonical vectors
- ```modifier_216_pairwise_obs_v128.py```: gradual 128:2:16 pruning, pair-wise version of canonical vectors

# Scripts examples (scripts/)
### Target sparsity 75%
- ```30epochs_gradual_pruning_squad_block8.sh```: original obs implementation adapted to 8-block pruning.

- ```oBERT28_squad.sh```: one-shot 2:8 pruning implementation.
- ```oBERT28_squad_gradual.sh```: gradual 2:8 pruning, m-combinatorial version of canonical vectors
- ```obs28_squad_gradual_pair.sh```: gradual 2:8 pruning, m-combinatorial version of canonical vectors

- ```oBERTnm8v64_squad.sh```: one-shot 64:2:8 pruning implementation
- ```oBERTnm8v128_squad.sh```: one-shot 128:2:8 pruning implementation

- ```oBERT28v64_squad_gradual.sh```: gradual 64:2:8 pruning, m-combinatorial version of canonical vectors
- ```oBERT28v128_squad_gradual.sh```: gradual 128:2:8 pruning, m-combinatorial version of canonical vectors

- ```obs28v64_squad_gradual_pair.sh```: gradual 64:2:8 pruning, pair-wise version of canonical vectors
- ```obs28v128_squad_gradual_pair.sh```: gradual 64:2:8 pruning, pair-wise version of canonical vectors
### Target sparsity 87.5%
- ```30epochs_gradual_pruning_squad_block8_875.sh```: original obs implementation adapted to 8-block pruning.
- ```oBERT216_squad.sh```: one-shot 2:16 pruning implementation.
- ```oBERT216_squad_gradual.sh```: gradual 2:16 pruning, m-combinatorial version of canonical vectors
- ```obs216_squad_gradual_pair.sh```: gradual 2:16 pruning, m-combinatorial version of canonical vectors


- ```oBERTnm16v64_squad.sh```: one-shot 64:2:16 pruning implementation
- ```oBERTnm16v128_squad.sh```: one-shot 128:2:16 pruning implementation

- ```oBERT216v64_squad_gradual.sh```: gradual 64:2:16 pruning, m-combinatorial version of canonical vectors
- ```oBERT216v128_squad_gradual.sh```: gradual 128:2:16 pruning, m-combinatorial version of canonical vectors

- ```obs216v64_squad_gradual_pair.sh```: gradual 64:2:16 pruning, pair-wise version of canonical vectors
- ```obs216v128_squad_gradual_pair.sh```: gradual 128:2:16 pruning, pair-wise version of canonical vectors
# Recipes examples (recipes/)
### Target sparsity 75%
- ```30epochs_8block75_squad.yaml```  original obs implementation adapted to 8-block pruning.
- ```oneshot_oBERT_28.yaml```: one-shot 2:8 pruning implementation.
- ```oBERT_28_gradual.yaml```: gradual 2:8 pruning, m-combinatorial version of canonical vectors
- ```obs28_gradual_pair.yaml```: gradual 2:8 pruning, pair-wise version of canonical vectors
- ```oneshot_oBERT_28v64.yaml```: one-shot 64:2:8 pruning implementation
- ```oneshot_oBERT_28v128.yaml```: one-shot 128:2:8 pruning implementation

- ```oBERT_28v64_gradual.yaml```: gradual 64:2:8 pruning, m-combinatorial version of canonical vectors
- ```oBERT_28v128_gradual.yaml```: gradual 128:2:8 pruning, m-combinatorial version of canonical vectors

- ```obs28v64_gradual_pair.yaml```: gradual 64:2:8 pruning, pair-wise version of canonical vectors
- ```obs28v128_gradual_pair.yaml```: gradual 128:2:8 pruning, pair-wise version of canonical vectors
### Target sparsity 87.5%
- ```30epochs_8block875_squad.yaml```  original obs implementation adapted to 8-block pruning.
- ```oneshot_oBERT_216.yaml```: one-shot 2:16 pruning implementation.
- ```oBERT_216_gradual.yaml```: gradual 2:16 pruning, m-combinatorial version of canonical vectors
- ```obs216_gradual_pair.yaml```: gradual 2:16 pruning, pair-wise version of canonical vectors

- ```oneshot_oBERT_216v64.yaml```: one-shot 64:2:16 pruning implementation
- ```oneshot_oBERT_216v128.yaml```: one-shot 128:2:16 pruning implementation

- ```oBERT_216v64_gradual.yaml```: gradual 64:2:16 pruning, m-combinatorial version of canonical vectors
- ```oBERT_216v128_gradual.yaml```: gradual 128:2:16 pruning, m-combinatorial version of canonical vectors

- ```obs216v64_gradual_pair.yaml```: gradual 64:2:16 pruning, pair-wise version of canonical vectors
- ```obs216v128_gradual_pair.yaml```: gradual 128:2:16 pruning, pair-wise version of canonical vectors