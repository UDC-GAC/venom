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

# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Modifier classes implementing the blockwise version of the Optimal Brain Surgeon
pruning framework, optimized for small blocks. The algorithm is described in details
in the Optimal BERT Surgeon paper https://arxiv.org/abs/2203.07259
"""
import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

import numpy as np

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer
from sparseml.pytorch.utils import GradSampler
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.utils import interpolate

from math import comb
import itertools
import torch.profiler
import numpy as np

__all__ = [
    "OBS216v128pairPruningModifier",
    "OBS216v128pairPruningParamsScorer",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class OBS216v128pairPruningModifier(BaseGradualPruningModifier):
    """
    As described in https://arxiv.org/abs/2203.07259
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given number of epochs.
    Uses the Optimal BERT Surgeon algorithm to prune weights based on the
    approximate second-order information of the loss function. When pruning,
    it also updates remaining weights to compensate for accuracy drops incurred
    by pruning. It follows the Optimal Brain Surgeon framework with optimizations
    to make it efficient but accurate for huge models.
    It can be used to prune other models besides BERT too.
    Naming convention with respect to the paper:
        - damp == small dampening constant 'lambda'
        - num_grads == number of gradient outer products 'm'
        - fisher_block_size == size of the blocks 'B' along the main diagonal
    Memory requirements: O(dB), where 'd' is the total number of prunable weights.
    If O(dB) can't fit on a single GPU device, pytorch DDP should be used to split
    the computational overhead equally between devices.
    Supported mask types: unstructured and block4.
    | Sample yaml:
    |   !OBSPruningModifier
    |       init_sparsity: 0.7
    |       final_sparsity: 0.9
    |       start_epoch: 2.0
    |       end_epoch: 26.0
    |       update_frequency: 4.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       mask_type: unstructured
    |       num_grads: 1024
    |       damp: 1e-7
    |       fisher_block_size: 50
    |       num_recomputations: 1
    |       grad_sampler_kwargs:
    |           batch_size: 8
    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch.
        Can also be a Dict of final sparsity values to a list of parameters to apply
        them to. If given a Dict, then params must be set to [] and the params to
        be pruned will be read from the final_sparsity Dict
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param mask_type: String to define type of sparsity to apply. 'unstructured'
        and 'block4' are supported. Default is 'unstructured'
    :param global_sparsity: set True to enable global pruning. If False, pruning will
        be layer-wise. Default is True
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    :param num_recomputations: number of recomputations of the Hessian approximation
        while performing one pruning step
    :param grad_sampler_kwargs: kwargs to override default train dataloader config
        for pruner's gradient sampling.
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        global_sparsity: bool = True,
        mask_type: str = "3:5",
        num_grads: int = 1024,
        damp: float = 1e-7,
        fisher_block_size: int = 50,
        grad_sampler_kwargs: Optional[Dict[str, Any]] = None,
        num_recomputations: int = 1,
    ):
        super().__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            inter_func=inter_func,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=global_sparsity,
            leave_enabled=leave_enabled,
            parent_class_kwarg_names=[],
        )
        self._mask_type = mask_type
        self._num_grads = int(num_grads)
        self._damp = damp
        self._fisher_block_size = int(fisher_block_size)
        self._grad_sampler_kwargs = grad_sampler_kwargs
        self._num_recomputations = num_recomputations
        self._last_applied_sparsity = 0.  # keep track for recomputations steps

        self._grad_sampler = None
        self._supported_masks = ("2:16", "3:16", "4:16", "5:16", "6:16", "7:16", "8:16", "9:16", "10:16", "11:16", "12:16", "6:8", "5:8", "4:8", "3:8", "2:8", "unstructured")

        self._validate()

    def _validate(self):
        if isinstance(self._damp, str):  # to support 'damp: 1e-7' in the recipe
            self._damp = float(self._damp)

        if self._mask_type not in self._supported_masks:
            raise ValueError(f"{self._mask_type} mask_type not supported")

    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type

    @ModifierProp()
    def num_grads(self) -> int:
        """
        :return: number of gradients used to calculate the Fisher approximation
        """
        return self._num_grads

    @ModifierProp()
    def damp(self) -> float:
        """
        :return: dampening factor used for inverse Fisher calculation
        """
        return self._damp

    @ModifierProp()
    def fisher_block_size(self) -> int:
        """
        :return: size of blocks along the main diagonal of the Fisher approximation
        """
        return self._fisher_block_size

    @ModifierProp()
    def num_recomputations(self) -> int:
        """
        :return: number of recomputations of the Hessian approximation during
            one pruning step
        """
        return self._num_recomputations

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.
        Expects `grad_sampler` dict with `data_loader_builder` and `loss_function`
        to initialize GradSampler instance and optionally override data-loader's
        hyperparams with `grad_sampler_kwargs` given in the recipe.
        :param module: the PyTorch model/module to modify
        :param epoch: the epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: optional list of loggers to log the modification process to
        :param kwargs: optional kwargs to support specific arguments
            for individual modifiers.
        """
        _LOGGER.info("Initializing OBS216v128pairPruningModifier")
        if (
            "grad_sampler" not in kwargs
            or "data_loader_builder" not in kwargs["grad_sampler"]
            or "loss_function" not in kwargs["grad_sampler"]
        ):
            raise RuntimeError(
                "grad_sampler dict with data_loader_builder and loss_function "
                "must be provided to initialize GradSampler"
            )

        self._grad_sampler = GradSampler(
            kwargs["grad_sampler"]["data_loader_builder"](self._grad_sampler_kwargs),
            kwargs["grad_sampler"]["loss_function"],
        )

        super().initialize(module, epoch, loggers, **kwargs)

    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        if steps_per_epoch == 1 and not math.isinf(epoch):
            return  # not a one-shot run

        torch.cuda.empty_cache()
        if self._scorer._is_main_proc:
            _LOGGER.info("Running OBS Pruning")
            self._scorer._enabled_grad_buffering = True

        self._pre_step_completed = True
        to_apply_sparsities = self.get_applied_sparsity_for_epoch(
            epoch, steps_per_epoch
        )
        last_applied_sparsities = (
            self._last_applied_sparsity
            if isinstance(self._last_applied_sparsity, List)
            else [self._last_applied_sparsity] * len(to_apply_sparsities)
        )

        for i in range(1, self._num_recomputations + 1):
            self._collect_grad_samples(module, self._grad_sampler)
            _LOGGER.info("Sparsity recomputation ...")
            recomputation_sparsity = [
                interpolate(
                    i,
                    0,
                    self._num_recomputations,
                    start_sparsity,
                    target_sparsity,
                )
                for start_sparsity, target_sparsity in zip(last_applied_sparsities, to_apply_sparsities)
            ]

            _LOGGER.info("check_mask_update "+str(epoch) + " " + str(self._num_recomputations) + " " + str(to_apply_sparsities))
            # overwrite sparsity targets when there are recomputations
            super().check_mask_update(
                module,
                epoch,
                steps_per_epoch,
                recomputation_sparsity=recomputation_sparsity,
            )

        torch.cuda.empty_cache()
        self._last_applied_sparsity = to_apply_sparsities
        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = False

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return get_mask_creator_default(self.mask_type)

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return OBS216v128pairPruningParamsScorer(
            params=params,
            num_grads=self._num_grads,
            damp=self._damp,
            fisher_block_size=self._fisher_block_size,
            mask_type=self._mask_type,
        )

    def _collect_grad_samples(
        self,
        module: Module,
        grad_sampler: GradSampler,
    ):
        if not isinstance(grad_sampler, GradSampler):
            raise ValueError(
                "One-shot OBS pruning requires a GradSampler object given by the "
                f"grad_sampler kwarg. Given an object of type {type(grad_sampler)}"
            )

        is_training = module.training
        _LOGGER.info("Setting the model in the eval mode")
        module.eval()

        _LOGGER.info(f"Starting to collect {self._num_grads} grads with GradSampler")
        for i in grad_sampler.iter_module_backwards(module, self._num_grads):
            self._module_masks.pre_optim_step_update()

        if is_training:
            _LOGGER.info("Setting the model back to the train mode")
            module.train()


class OBS216v128pairPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters using the equations introduced in the Optimal BERT Surgeon
    to solve for the optimal weight update in the Optimal Brain Surgeon (OBS)
    framework. Implements unstructured and semi-structured (block4) scoring and
    pruning.
    :param params: list of model Parameters to track and score
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    """

    def __init__(
        self,
        params: List[Parameter],
        num_grads: int,
        damp: float,
        fisher_block_size: int,
        mask_type: str,
    ):
        super().__init__(params)
        self._num_grads = num_grads
        self._damp = damp
        self._fisher_block_size = fisher_block_size
        self._mask_type = mask_type

        self._finvs = None  # type: List[EmpiricalBlockFisherInverse]
        self._enabled_grad_buffering = False
        self._eps = torch.finfo(torch.float32).eps

        # assign device to each Finv
        self._devices = []
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            self._devices = [torch.device("cpu")] * len(self._params)
        else:
            num_devices = min(num_devices, len(self._params))
            per_device = math.floor(len(self._params) / num_devices)
            for i in range(num_devices):
                self._devices += [torch.device("cuda", i)] * per_device
            remainder = len(self._params) - len(self._devices)
            if remainder > 0:
                self._devices += [self._devices[-1]] * remainder

        self._pickle_exclude_params.extend(
            [
                "_finvs",
                "_enabled_grad_buffering",
                "_devices",
            ]
        )
        self._validate()

    @torch.no_grad()
    def solve(self, set_a, set_b, m, n, indices_a, indices_b, tensor_shape, device_, dtype_):
        best_selected_a = set_a[:, :m]
        best_indices_a = indices_a[:, :m]

        #print(indices_b.shape[1], best_indices_a.shape[1])
        best_indices_b = torch.tensor([], dtype=indices_b.dtype)
        if indices_b.shape[1]!=0:
            max_row_length = indices_b.shape[1] - best_indices_a.shape[1]

            A = indices_b.reshape(indices_b.shape[0], indices_b.shape[1], 1)
            B = best_indices_a.reshape(best_indices_a.shape[0], 1, best_indices_a.shape[1])
            subset_indices_b = ~(A == B).sum(-1).to(dtype=bool)

            subset_b = set_b[subset_indices_b].reshape(-1, max_row_length)
            sub_indices_b = indices_b[subset_indices_b].reshape(-1, max_row_length)
            best_selected_b = subset_b[:, :n]
            best_indices_b = sub_indices_b[:, :n]

            value = torch.sum(torch.cat((best_selected_a, best_selected_b), dim=1), dim=1).to(dtype=dtype_)

        else:
            value = torch.sum(best_selected_a, dim=1).to(dtype=dtype_)

        return value, best_indices_a, best_indices_b

    @torch.no_grad()
    def bool_to_index(self, tensor_bool):
        return torch.nonzero(tensor_bool).squeeze()

    @torch.no_grad()
    def find_best_combination0(self, scores, mm_row, nn_row, tensor_shape, device_, dtype_):
        min_value = torch.ones(scores.shape[0])*torch.inf

        best_indices_a2 = torch.zeros(tensor_shape).bool()
        best_indices_b2 = torch.zeros(tensor_shape[0], tensor_shape[1]//2).bool()

        new_set_a = scores[:, :mm_row//2]
        new_set_b = scores[:, mm_row//2:]

        set_a, indices_a = torch.sort(new_set_a, 1)
        set_b, indices_b = torch.sort(new_set_b, 1)

        a=nn_row//2
        b=nn_row-a*2
        for i in range(nn_row//2+1):
            if (mm_row//2-a) >= b:
                value, ind_a, ind_b = self.solve(set_a, set_b, a, b, indices_a, indices_b, tensor_shape, "cpu", dtype_)

                lt = torch.lt(value, min_value)

                if lt.any():
                    min_value[lt] = value[lt]

                    tmp = torch.zeros((lt.sum().item(), mm_row)).bool()
                    idx = ind_a[lt,:]*2
                    idx = torch.cat((idx, idx+1), dim=1)
                    tmp = tmp.scatter(1,idx,True)
                    best_indices_a2[lt] = tmp

                    tmp = torch.zeros((lt.sum().item(), mm_row//2)).bool()
                    idx = ind_b[lt,:]
                    tmp = tmp.scatter(1, idx, True)
                    best_indices_b2[lt] = tmp
            a-=1; b+=2

        return best_indices_a2, best_indices_b2

    @torch.no_grad()
    def find_best_combination1(self, scores, mm_row, nn_row, tensor_shape, device_, dtype_, v=None, exclude_bool=None, best_to_prune_nm_idx=None, m_p=0):
        min_value = torch.ones(scores.shape[0])*torch.inf
        best_indices_a2 = torch.zeros(tensor_shape).bool()
        best_indices_b2 = torch.zeros(tensor_shape).bool()

        set_a = scores[:, :mm_row//2]
        set_b = scores[:, mm_row//2:]

        if exclude_bool is not None:
            tensor_and = torch.logical_and(exclude_bool[:, :, 0], exclude_bool[:, :, 1])
            tensor_and = tensor_and.view(tensor_and.shape[0], -1)
            tensor_xor = torch.logical_xor(exclude_bool[:, :, 0], exclude_bool[:, :, 1])
            tensor_xor = tensor_xor.view(tensor_xor.shape[0], -1)
            tensor_or = torch.logical_or(exclude_bool[:, :, 0], exclude_bool[:, :, 1])
            tensor_or = tensor_or.view(tensor_or.shape[0], -1)
            #print("tensor_and", tensor_and[:2])

            selected_values = set_a[~tensor_or]
            new_set_a = torch.ones_like(set_a)*torch.inf
            new_set_a[~tensor_or] = selected_values

            selected_values = set_a[tensor_xor]
            remaining_b = torch.zeros_like(set_a)
            remaining_b[tensor_xor] = selected_values
            remaining_b_rows_a, remaining_b_cols_a = remaining_b.nonzero(as_tuple=True)
            plus = best_to_prune_nm_idx[remaining_b_rows_a, remaining_b_cols_a]
            remaining_b_cols_a2 = remaining_b_cols_a*2+(plus+1)%2

            selected_values = set_b[~tensor_or]
            new_set_b = torch.ones_like(set_b)*torch.inf
            new_set_b[~tensor_or] = selected_values
            new_set_b[tensor_xor] = remaining_b[tensor_xor]

            remaining_cols_b = torch.zeros(new_set_b.shape, dtype=remaining_b_rows_a.dtype)+torch.tensor(range(0,new_set_b.shape[1]))
            remaining_cols_b = remaining_cols_b*2+best_to_prune_nm_idx
            remaining_cols_b[remaining_b_rows_a, remaining_b_cols_a] = remaining_b_cols_a2

            del tensor_and
            del tensor_xor
            del tensor_or
            del selected_values
            del remaining_b_cols_a2
            del remaining_b_rows_a
            del remaining_b_cols_a

            new_nn_row = nn_row-m_p

        else:
            new_set_a = set_a
            new_set_b = set_b
            new_nn_row = nn_row

        set_a, indices_a = torch.sort(new_set_a, 1)
        set_b, indices_b = torch.sort(new_set_b, 1)

        a=new_nn_row//2
        b=new_nn_row-a*2
        for i in range(new_nn_row//2+1):
            if (mm_row//2-a) >= b:
                value, ind_a, ind_b = self.solve(set_a, set_b, a, b, indices_a, indices_b, tensor_shape, "cpu", dtype_)

                lt = torch.lt(value, min_value)
                notinf = value<torch.inf

                lt = lt&notinf
                if lt.any():
                    min_value[lt] = value[lt]

                    tmp = torch.zeros((lt.sum().item(), mm_row)).bool()
                    #idx = torch.gather(remaining_cols_a[lt], 1, ind_a[lt,:])
                    idx = ind_a[lt,:]*2
                    idx = torch.cat((idx, idx+1), dim=1)
                    tmp = tmp.scatter(1,idx,True)
                    best_indices_a2[lt] = tmp

                    tmp = torch.zeros((lt.sum().item(), mm_row)).bool()
                    idx = torch.gather(remaining_cols_b[lt], 1, ind_b[lt,:])
                    tmp = tmp.scatter(1, idx, True)
                    best_indices_b2[lt] = tmp

            a-=1; b+=2
        return best_indices_a2, best_indices_b2

    @torch.no_grad()
    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored based on the blockwise OBS
        """
        scores = [None] * len(self._params)
        block_finv_w = [None] * len(self._params)

        # Change depending on your n:m pattern and V value. Examples
        #nm = {0.0:(3,8), 0.375:(4,8), 0.5:(5,8), 0.625:(6,8)}
        nm = {0.0:(7,16), 0.4375:(8,16), 0.5:(9,16), 0.5625:(10,16), 0.625:(11,16), 0.6875:(12,16), 0.75:(13,16), 0.8125:(14,16)}
        #nm = {0.0:(10,16), 0.625:(11,16), 0.6875:(12,16), 0.75:(13,16), 0.8125:(14,16)}
        #nm = {0.0:(8,16), 0.4375:(9,16), 0.5:(10,16), 0.5625:(11,16), 0.625:(12,16), 0.6875:(13,16), 0.75:(14,16)}

        sparsity = self._last_applied_sparsity
        nn_row, mm_row = nm[sparsity]
        #nn_row, mm_row = self._mask_type.split(":")
        #mm_row=int(mm_row); nn_row=mm_row-int(nn_row)

        m=2; n=1
        v=128
        m_p = (nn_row-2) if ((mm_row-nn_row)>3) else (mm_row-4)

        _LOGGER.info("_last_applied_sparsity "+ str(nn_row) + " " + str(mm_row) + " " + str(sparsity))

        if self._is_main_proc:
            for i, finv in enumerate(self._finvs):
                block_w = self._params[i].data.view(-1, m).to("cpu")
                finv_cpu = finv.f_inv.to("cpu")
                block_finv = (
                    torch.cat(
                        [
                            finv_cpu[:, i : i + m, i : i + m]
                            for i in range(0, finv.B, m)
                        ],
                        dim=1,
                    )
                    .reshape((finv.d // finv.B, finv.B // m, m, m))
                    .reshape((finv.d // m, m, m))
                )

                _LOGGER.info("layer " + str(i) + " (out of " + str(len(self._finvs)) + "). Device cpu")
                nrows, ncols = self._params[i].data.shape
                _LOGGER.info("nrows: "+str(nrows)+", ncols: "+str(ncols)+", v: "+str(v)+", m_p: "+str(m_p) )

                ############ pair-wise
                block_finv_w_pair = torch.linalg.solve(
                        block_finv,
                        block_w,
                    )
                scores_pair = 0.5 * torch.einsum(
                    "bi,bi->b", block_w, block_finv_w_pair
                )

                ############ indv-wise

                e_nm = torch.tensor([
                            [1, 0],
                            [0, 1]
                        ], dtype=torch.long)
                num_combs = e_nm.shape[0]

                ee_nm = torch.einsum("bij,bkj->bik", e_nm.view(num_combs,m,1), e_nm.view(num_combs,m,1))

                best_score_nmv = torch.ones((nrows//v, ncols//m))*torch.inf
                best_score_nm = torch.ones(block_w.shape[0])*torch.inf
                best_to_prune_idx = torch.zeros((nrows//v, ncols//m), dtype=torch.long)
                best_block_finv_w = torch.zeros((block_w.shape[0], n))

                for k in range(num_combs):
                    #block_w_nm = block_w[:, e_nm[k].nonzero().squeeze()]
                    #block_finv_nm = block_finv[:,ee_nm[k].ne(0)].view(-1,n,n)
                    block_w_nm = torch.masked_select(block_w, e_nm[k].unsqueeze_(0).bool()).view(-1, n)
                    block_finv_nm = torch.masked_select(block_finv, ee_nm[k].unsqueeze_(0).bool()).view(-1, n, n)

                    # cache finv_w products for OBS weight update
                    block_finv_w_nm = torch.linalg.solve(
                        block_finv_nm,
                        block_w_nm,
                    )  # (d/4, 2)

                    score_nm = 0.5 * torch.einsum(
                        "bi,bi->b", block_w_nm, block_finv_w_nm
                    )  # d/4

                    val = torch.stack([torch.sum(s,0) for s in torch.split(score_nm.view(-1, ncols//m), v)])

                    lt = torch.lt(val, best_score_nmv)
                    if lt.any():
                        best_score_nmv = torch.where(lt, val, best_score_nmv)
                        best_to_prune_idx[lt] = k
                        w = lt.unsqueeze(1).repeat(1,v,1).flatten()
                        best_block_finv_w[w,:] = block_finv_w_nm[w, :]
                        best_score_nm = torch.where(w, score_nm, best_score_nm)

                    if i==3 or i==0:
                        _LOGGER.info("block_finv_w_nm "+str(block_finv_w_nm.reshape(-1,mm_row//2)[:4]))#FIXME: block_finv w(1,1) where w= 0,x != w(0,1) where w=0,x

                ############ choose best
                best_indv_scores = best_score_nm.view(-1, mm_row//2)
                best_pair_scores = scores_pair.view(-1, mm_row//2)

                #best_pair_scores+=torch.inf
                #best_indv_scores+=torch.inf

                ############ choose best mm_row-m_p v columns
                scores_indv_v = torch.stack([torch.sum(s,0) for s in torch.split(best_indv_scores.view(-1, ncols//m), v)]).view(-1, mm_row//2)

                scores_pair_v = torch.stack([torch.sum(s,0) for s in torch.split(best_pair_scores.view(-1, ncols//m), v)]).view(-1, mm_row//2)

                sets = torch.cat((scores_pair_v, scores_indv_v), dim=1).reshape(-1, mm_row)

                best_to_prune_nm_idx = best_to_prune_idx.view(-1, mm_row//2)
                tensor_shape = (nrows//v*ncols//mm_row, mm_row)

                exclude_columns_a, exclude_columns_b = self.find_best_combination0(sets, mm_row, m_p, tensor_shape, "cpu", block_finv_w_pair.dtype)

                exclude_row_a, exclude_col_a = exclude_columns_a.nonzero(as_tuple=True)
                exclude_row_b, exclude_col_b = exclude_columns_b.nonzero(as_tuple=True)
                plus = best_to_prune_nm_idx[exclude_row_b, exclude_col_b]
                exclude_col_b = exclude_col_b*2+plus

                exclude_row = torch.cat((exclude_row_a, exclude_row_b))
                exclude_col = torch.cat((exclude_col_a, exclude_col_b))

                block_finv_w_pair = block_finv_w_pair.reshape(-1,mm_row)
                block_finv_w_nflatten = best_block_finv_w.view(-1,mm_row//2)

                ############ initialize selected weights in block_finv_w
                block_finv_w0 = torch.zeros((nrows*ncols//mm_row, mm_row), dtype=block_finv_w_pair.dtype)
                scores0 = torch.zeros(block_finv_w0.shape, dtype=block_finv_w_pair.dtype)

                exclude_columns_a = exclude_columns_a.reshape(-1, ncols).unsqueeze(1).repeat(1,v,1).reshape(-1,mm_row)
                exclude_columns_b = exclude_columns_b.reshape(-1, ncols//2).unsqueeze(1).repeat(1,v,1).reshape(-1,mm_row//2)
                best_to_prune_nm_idx = best_to_prune_nm_idx.view(-1, ncols//2).unsqueeze(1).repeat(1,v,1).view(-1, mm_row//2)

                block_finv_w0[exclude_columns_a] = block_finv_w_pair[exclude_columns_a]
                scores0[exclude_columns_a] = -1

                val = block_finv_w_nflatten[exclude_columns_b]
                rows_b, cols_b = exclude_columns_b.nonzero(as_tuple=True)
                plus = best_to_prune_nm_idx[rows_b, cols_b]
                block_finv_w0[rows_b,cols_b*2+plus] = val
                scores0[rows_b,cols_b*2+plus] = -1
                ############ Update remaining weights
                tensor_shape = (nrows*ncols//mm_row, mm_row)

                cols_to_exclude = m_p*(ncols//mm_row)
                _, indices = torch.sort(exclude_row, descending=False, stable=True)

                exclude = exclude_col[indices].reshape(-1, cols_to_exclude).unsqueeze(1).repeat(1,v,1).reshape(-1, m_p)
                exclude_bool = torch.zeros(tensor_shape).bool()
                exclude_bool = exclude_bool.scatter(1,exclude,True)
                exclude_bool = exclude_bool.view(exclude_bool.shape[0], -1, 2)
                del indices
                del exclude

                tensor_xor = torch.logical_xor(exclude_bool[:, :, 0], exclude_bool[:, :, 1])
                tensor_xor_reshaped = tensor_xor.view(-1, mm_row//2)

                remaining_b_rows_a, remaining_b_cols_a = tensor_xor_reshaped.nonzero(as_tuple=True)
                plus = best_to_prune_nm_idx[remaining_b_rows_a, remaining_b_cols_a]
                remaining_b_cols_a = remaining_b_cols_a*2+(plus+1)%2
                block_finv_w_nflatten[tensor_xor_reshaped] = block_finv_w_pair[remaining_b_rows_a, remaining_b_cols_a]

                ############ choose best remaining v columns
                sets = torch.cat((best_pair_scores, best_indv_scores), dim=1).reshape(-1, mm_row)

                best_indices_a, best_indices_b = self.find_best_combination1(sets, mm_row, nn_row, tensor_shape, "cpu", block_finv_w_pair.dtype, v, exclude_bool, best_to_prune_nm_idx, m_p)

                block_finv_w0[best_indices_a] = block_finv_w_pair[best_indices_a]

                tensor_bool_reshaped = best_indices_b.view(block_finv_w0.shape[0], -1, 2)
                tensor_xor = torch.logical_xor(tensor_bool_reshaped[:, :, 0], tensor_bool_reshaped[:, :, 1])
                tensor_xor_reshaped = tensor_xor.view(tensor_xor.shape[0], -1)
                block_finv_w0[best_indices_b] = block_finv_w_nflatten[tensor_xor_reshaped]

                block_finv_w[i] = block_finv_w0.to(self._devices[i])

                if i==3 or i==0:
                    _LOGGER.info("block_w "+str(block_w.view(-1, mm_row)[:4]))
                    _LOGGER.info("block_finv_w_pair "+str(block_finv_w_pair[:4]))
                    _LOGGER.info("best_to_prune_idx "+str(best_to_prune_idx.view(-1, mm_row//2)[:4]))
                    _LOGGER.info("sets "+str(sets[:4]))

                #scores[i] = (block_finv_w[i].view(self._params[i].shape) != 0) * -1.0 #FIXME:
                scores0[best_indices_a] = -1.0
                scores0[best_indices_b] = -1.0

                scores[i] = scores0.view(self._params[i].shape).to(self._devices[i])

                # make sure pruned ones will stay pruned
                #for i, score in enumerate(scores):
                if self._masks[i].all():
                    mask = (self._params[i].data==0)
                    scores[i][mask] = float("-inf")
                else:
                    scores[i][self._masks[i] == 0] = float("-inf")

                if i==3 or i==0:
                    _LOGGER.info("block_finv_w "+str(block_finv_w[i].reshape(-1,mm_row)[:4]))
                    _LOGGER.info("scores "+str(scores[i].reshape(-1,mm_row)[:4]))

                """ for idx, subm in enumerate(scores[i].reshape(-1, 1, mm_row)):
                        total = torch.sum((~subm.any(axis=0)))
                        if(total!=(mm_row-nn_row)):
                            print(mm_row-nn_row)
                            _LOGGER.info("assert " + str(total)+" in "+str(idx)+"="+str(torch.sum((~subm.any(axis=0))))) """

                """ subm = scores[i].reshape(-1, 1, mm_row)
                wrong = (torch.sum((~subm.any(axis=1)), dim=1) != (mm_row-nn_row)).any()
                if wrong:
                    _LOGGER.info(str(torch.sum((~subm.any(axis=1)), dim=1)))
                assert (not wrong) """

                ##check format
                """ matrix = scores[i].view((nrows,ncols)).to("cpu").numpy()
                input_shape = matrix.shape
                rows, cols = input_shape[0], input_shape[1]
                d_rows, d_cols = v, mm_row
                subm_rows, subm_cols = rows-d_rows+1, cols-d_cols+1
                ii, jj = np.meshgrid(range(0, subm_rows, d_rows), range(0, subm_cols, d_cols), indexing='ij')
                d_ii, d_jj = np.meshgrid(range(d_rows), range(d_cols), indexing='ij')
                subm_ii = ii[:, :, np.newaxis, np.newaxis] + d_ii
                subm_jj = jj[:, :, np.newaxis, np.newaxis] + d_jj
                subm = matrix[subm_ii, subm_jj]

                for i,s in enumerate((subm==0).reshape(-1,v,mm_row)):
                    assert ( not (np.sum((~s.any(axis=0))) < m_p) ) """
                ##

                #break
            # make sure pruned ones will stay pruned
            """ for i, score in enumerate(scores):
                score[self._masks[i] == 0] = float("-inf") """

        self._broadcast_list_from_main(scores)
        self._broadcast_list_from_main(block_finv_w)
        self._block_finv_w = block_finv_w  # cache for OBS weight update

        return scores

    @torch.no_grad()
    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Update the empirical inverse Fisher estimation based on the current gradients
        :param masks: latest masks that are applied to these parameters
        """
        #_LOGGER.info(f"pre_optim_step_update() method")
        if not self._enabled_grad_buffering:
            # only collect gradients when called during pruning step
            # this ignores calls invoked by manager during training
            return

        #_LOGGER.info(f"Setup Fisher masks")
        if self._finvs is None:
            self._setup_fisher_inverse(masks)

        #_LOGGER.info(f"Update empirical inverse Fisher estimation")
        for i, finv in enumerate(self._finvs):
            self._params[i].grad.mul_(masks[i])
            finv.add_grad(self._params[i].grad.view(-1).to(self._devices[i]))

    @torch.no_grad()
    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Apply OBS weight update which zeros-out pruned weights and updates the
        remaining weights to preserve the loss.
        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """
        obs_updates = [None] * len(self._params)
        if self._is_main_proc:
            for i, param in enumerate(self._params):
                _LOGGER.info("assert in " + str(i))
                #print("assert in ", i, "mask_diffs[i]", mask_diffs[i].shape, "self._block_finv_w[i]", self._block_finv_w[i].shape)
                #print( torch.all((mask_diffs[i] == -1) == (self._block_finv_w[i].view(mask_diffs[i].shape) != 0)) )

                """ ok = str(torch.all((mask_diffs[i].to(self._devices[i]) == -1) == (self._block_finv_w[i].view(mask_diffs[i].shape) != 0)))
                if not ok:
                    _LOGGER.info(str(mask_diffs[i].to(self._devices[i]).reshape(-1,8)[:4]))
                    _LOGGER.info(str(self._block_finv_w[i].reshape(-1,8)[:4])) """

                #assert torch.all((mask_diffs[i] == -1) == (self._block_finv_w[i].view(mask_diffs[i].shape) != 0))

                if i==0 or i==3:
                    _LOGGER.info("mask df "+str(mask_diffs[i].reshape(-1,16)[:4]))

                #obs_updates[i] = (
                #    self._finvs[i]
                #    .mul(self._block_finv_w[i].view(-1))
                #    .view(param.data.shape)
                #)
                obs_updates[i] = (
                        self._finvs[i]
                        .mul(
                            self._block_finv_w[i].view(-1)
                            * (mask_diffs[i].to(self._devices[i]) == -1).view(-1).to(self._devices[i])
                        )
                        .view(param.data.shape)
                )

                #+ self._eps)
                #obs_updates[i] = (
                #        self._finvs[i]
                #        .mul(
                #            (param.data * (mask_diffs[i] == -1))
                #            .view(-1)
                #            .to(self._devices[i])
                #            / (self._finvs[i].diag() )
                #        )
                #        .view(param.data.shape)
                #)

        self._broadcast_list_from_main(obs_updates)
        # apply OBS update and manually zero-out pruned weights
        for i, param in enumerate(self._params):
            param.data -= obs_updates[i].to(param.data.device)
            param.data[mask_diffs[i] == -1] = 0.0

        self._finvs = None

    def _validate(self):
        if self._mask_type == "block4":
            for param in self._params:
                assert (
                    param.numel() % self._fisher_block_size == 0
                ), "number of elements in each param must be divisible by fisher_block_size"

    def _setup_fisher_inverse(self, masks: List[Tensor]):
        self._masks = masks  # to be used by score_parameters
        self._finvs = []
        for i, param in enumerate(self._params):
            self._finvs.append(
                EmpiricalBlockFisherInverse(
                    self._num_grads,
                    self._fisher_block_size,
                    param.numel(),
                    self._damp,
                    self._devices[i],
                )
            )


class EmpiricalBlockFisherInverse:
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device

        self.num_blocks = math.ceil(self.d / self.B)
        self.f_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # O(d x B) memory

    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched f_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        finv_g = torch.einsum("bij,bj->bi", self.f_inv, g)

        # scalar denominator for each batch: (batch)
        alpha = (self.m + torch.einsum("bi,bi->b", g, finv_g)).sqrt().unsqueeze(1)
        finv_g /= alpha

        # update f_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.f_inv.baddbmm_(finv_g.unsqueeze(2), finv_g.unsqueeze(1), alpha=-1)

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.f_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.f_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]
