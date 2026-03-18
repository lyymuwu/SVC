import os
import pickle
import time
import torch
from tqdm import tqdm
from src.ties_merging_utils import check_parameterNamesMatch, check_state_dicts_equal, ties_merging
from src.ties_merging_utils import state_dict_to_vector, vector_to_state_dict
from memory_profiler import profile


def WA(task_vector_avg, task_vectors, config):
    config.scaling_coef = 1
    return task_vector_avg

def TA(task_vector_avg, task_vectors, config):
    # config.scaling_coef = 0.3
    for key in task_vector_avg.vector:
        Wm = task_vector_avg.vector[key] * len(task_vectors) * 0.3
        task_vector_avg.vector[key] = Wm
    return task_vector_avg

def SA(task_vector_avg, task_vectors, config):
    config.scaling_coef = 1
    for key in task_vector_avg.vector:
        if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key and "token_embedding" not in key:
            Wm = task_vector_avg.vector[key] * len(task_vectors)
            task_vector_avg.vector[key] = Wm
    return task_vector_avg

def TIES(task_vector_avg, task_vectors, config):
    ft_checks = []
    for dataset_name in config.DATASETS:
        finetuned_checkpoint = os.path.join(config.base_dir, "checkpoints", config.model, dataset_name, "finetuned.pt")
        try:
            tmp = torch.load(
                finetuned_checkpoint, 
                weights_only=False,
            ).state_dict()
        except RuntimeError:
            tmp = pickle.load(
                open(finetuned_checkpoint, 'rb')
            ).cpu().state_dict()
        ft_checks.append(tmp)
    
    ptm_check = torch.load(config.pretrained_checkpoint, weights_only=False).state_dict()
    check_parameterNamesMatch(ft_checks + [ptm_check])
    
    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft.cpu() - flat_ptm.cpu()
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])for i in range(len(ft_checks))])


    K = 20
    merge_func = "dis-sum"
    config.scaling_coef_ = 0.3

    merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)
    merged_state_dict = vector_to_state_dict(merged_tv, ptm_check, remove_keys=remove_keys)
    task_vector_avg.vector = merged_state_dict
    return task_vector_avg



def layer_wise_TIES(task_vector_avg, task_vectors, config):
    """
    Layer-wise TIES merging function.
    This function merges task vectors layer by layer using TIES method.
    """
    for key in task_vectors[0].vector:
        Wts = torch.stack([tv.vector[key] for tv in task_vectors]) # [T, M, N]
        W_flat = Wts.reshape(len(Wts), -1)

        # Apply TIES merging on the current layer
        merged_layer = ties_merging(
            W_flat,
            reset_thresh=20,  # K value
            merge_func="dis-mean",  # or "dis-sum"
        )
        # Reshape back to original layer shape
        task_vector_avg.vector[key] = merged_layer.reshape(Wts.shape[1:])  

    return task_vector_avg



def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


def DARE(task_vector_avg, task_vectors, config):
    with torch.no_grad():
        drop_p = 0.8
        for key in task_vectors[0].vector:
            Wts = [task_vector.vector[key] for task_vector in task_vectors]
            Wts = torch.stack(Wts)
            
            Wts = mask_input_with_mask_rate(Wts, drop_p, True, "random")
            Wm = Wts.mean(dim=0)
            task_vector_avg.vector[key] = Wm
            
    return task_vector_avg


def TSVM(task_vector_avg, task_vectors, args): # task_vector_sum.vector, task_vectors
    print("Weight Decomposition...")
    for key, value in tqdm(task_vector_avg.vector.items()):
        if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key and "embedding" not in key:
            pass
        else:
            continue
        
        device = task_vectors[0].vector[key].device
        sv_reduction = 1 / len(task_vectors)
        for i, task_vector in enumerate(task_vectors):
            vec = task_vector.vector[key].cuda()  
            u, s, v = torch.linalg.svd(vec, full_matrices=False)
            
            if i == 0:
                print(f"Computed SVD for {key}...")
                sum_u = torch.zeros_like(u, device='cuda')
                sum_s = torch.zeros_like(s, device='cuda')
                sum_v = torch.zeros_like(v, device='cuda')
            reduced_index_s = int(s.shape[0] * sv_reduction) 
            
            # select only the first reduced_index_s columns of u and place them
            cur_len = sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s].shape[1]
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                :, :cur_len
            ]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                :cur_len
            ]
            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                :cur_len, :
            ]
        
        try:
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        except:
            noise = torch.eye(sum_u.shape[0], sum_u.shape[1]) * 1e-10
            u_u, s_u, v_u = torch.linalg.svd(sum_u+noise, full_matrices=False)
            noise = torch.eye(sum_v.shape[0], sum_v.shape[1]) * 1e-10
            u_v, s_v, v_v = torch.linalg.svd(sum_v+noise, full_matrices=False)
        
        U_tsm, V_tsm, S_tsm = u_u @ v_u, u_v @ v_v, sum_s
        W_TSM = U_tsm @ torch.diag(S_tsm) @ V_tsm
        task_vector_avg.vector[key] = W_TSM.to(device) #*2.4 #* 1.4
    
    return task_vector_avg


def ISO_C(task_vector_avg, task_vectors, config):
    print("Computing SVD...")
    with torch.no_grad():
        for key in task_vectors[0].vector:
            Wts = [task_vector.vector[key] for task_vector in task_vectors]
            task_vector_avg.vector[key] = sum(Wts) / len(Wts)
            device = task_vector_avg.vector[key].device

            if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
                Wm = (task_vector_avg.vector[key] * len(Wts)).cuda()
                Wts = torch.stack(Wts).cuda()
                U, S, V = torch.linalg.svd(Wm, full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                W = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_mean   ),
                        V,
                    )
                )
                task_vector_avg.vector[key] = W.to(device)

    return task_vector_avg


@torch.no_grad()
def ISO_CTS(task_vector_avg, task_vectors, config):
    common_space_fraction = 0.8
    print("Computing SVD...")
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape

        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key]
                if i == 0:
                    task_vector_avg.vector[key] = vec.clone()
                else:
                    task_vector_avg.vector[key] += (vec - task_vector_avg.vector[key]) / (i + 1)
            continue
        
        print(f"Computing common space using sum for {key}...")
        combined_w = sum([task_vector.vector[key] for task_vector in task_vectors])

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * common_space_fraction)
        _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(config.DATASETS)) * len(config.DATASETS)
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################
        
        ### Calculate task specific space ###
        n_dims_per_task = int((min(shape_) - common_space_index_s) / len(config.DATASETS))
        for i, task_vector in enumerate(task_vectors):
            w = task_vector.vector[key]

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            try:
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False) 
            except:
                noise = torch.eye(w_ts.shape[0], w_ts.shape[1]) * 1e-8
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts+noise, full_matrices=False)     
            
            if i == 0:
                combined_space_u = torch.zeros_like(u_ts)
                combined_space_s = torch.zeros_like(s_ts)
                combined_space_v = torch.zeros_like(v_ts)
                
            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
        ###################################################################
        
        combined_space_u[:, len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_u
        combined_space_s[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_s
        combined_space_v[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s, :] = common_space_v
        
        ### Orthogonalize combined_space_u and combined_space_v ###
        try:
            u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
            u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
        except:
            noise = torch.eye(combined_space_u.shape[0], combined_space_u.shape[1]) * 1e-8
            u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u + noise, full_matrices=False)
            noise = torch.eye(combined_space_v.shape[0], combined_space_v.shape[1]) * 1e-8
            u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v + noise, full_matrices=False)
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################
        
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                
        task_vector_avg.vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        ) #* 1.4
    
    return task_vector_avg



def _get_cfg(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)

@torch.no_grad()
def STAR(task_vector_avg, task_vectors, config):
    """
    Paper-aligned STAR (NAACL 2025):
      per-task SVD -> adaptive spectral truncation (eta%) -> nuclear-norm rescale -> reconstruct
      then simple average across tasks.

    Args:
        task_vector_avg: a TaskVector-like object with `.vector` dict (used as output container).
        task_vectors: list of TaskVector-like objects with `.vector` dict (per-task deltas).
        config: has `eta` (percent, e.g., 40) and optional `eps`, `known_rank` (ignored for correctness).
    Returns:
        task_vector_avg (in-place updated) as merged task vector.
    """
    eta = float(_get_cfg(config, "eta", 40.0))          # percent in [0, 100]
    eps = float(_get_cfg(config, "eps", 1e-12))
    # known_rank is for speed in LoRA setting; correctness does not depend on it.
    _ = _get_cfg(config, "known_rank", None)

    frac = max(0.0, min(1.0, eta / 100.0))

    def star_one_matrix(A: torch.Tensor) -> torch.Tensor:
        # STAR is defined on matrices; for >2D tensors, flatten to 2D outside.
        orig_dtype = A.dtype
        A32 = A.float()

        # Full SVD (paper definition)
        # torch.linalg.svd returns U, S, Vh where A = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(A32, full_matrices=False)

        total = S.sum()
        if total <= eps:
            return torch.zeros_like(A32, dtype=orig_dtype)

        # rank_keep: smallest r s.t. cumsum(S)/sum(S) >= eta%
        cdf = torch.cumsum(S, dim=0) / total
        # searchsorted gives first index where cdf[idx] >= frac
        r = int(torch.searchsorted(cdf, torch.tensor(frac, device=cdf.device), right=False).item()) + 1
        r = max(1, min(r, S.numel()))

        S_keep = S[:r]
        keep_sum = S_keep.sum()
        if keep_sum <= eps:
            # degenerate: keep part has ~0 nuclear norm
            return torch.zeros_like(A32, dtype=orig_dtype)

        # rescale to restore nuclear norm (||S||_1)
        scale = (total / keep_sum)
        S_rescaled = S_keep * scale

        # reconstruct: U[:, :r] @ diag(S_rescaled) @ Vh[:r, :]
        # do diag multiplication efficiently
        A_out = (U[:, :r] * S_rescaled.unsqueeze(0)) @ Vh[:r, :]
        return A_out.to(orig_dtype)

    # Apply STAR per task, then average (this is the key correction vs "SVD on Wm")
    for key in task_vector_avg.vector:
        W_list = [tv.vector[key] for tv in task_vectors]
        W0 = task_vector_avg.vector[key]

        # Bias / LayerNorm weight etc. (1D): STAR paper is matrix-based; keep simple average.
        if W0.ndim < 2:
            task_vector_avg.vector[key] = torch.stack(W_list, dim=0).mean(dim=0)
            continue

        # Flatten >2D (e.g., conv kernels) to 2D in a consistent way, then reshape back.
        orig_shape = W0.shape
        def to_2d(W):
            return W.reshape(orig_shape[0], -1) if W.ndim > 2 else W

        W_out_list = []
        for W in W_list:
            W2d = to_2d(W)
            W2d_out = star_one_matrix(W2d)
            W_out_list.append(W2d_out.reshape(orig_shape) if W.ndim > 2 else W2d_out)

        task_vector_avg.vector[key] = torch.stack(W_out_list, dim=0).mean(dim=0)

    return task_vector_avg



@torch.no_grad()
def coef_cal_multi_ranks(Wts, Wm, Um, Sm, Vm, k, row_space=False, alpha=1.0, target=-1):
    """
    Parallel computation of calibration coefficients for multiple ranks.

    If target != -1, only compute projections for the task at index `target`,
    and compute eta per-rank using that single task's s-value:
      - if s[:, target] > 0: eta = max(s[:, target], alpha)
      - else: eta = 1.0

    This keeps the original logic when target == -1.
    """
    n_ranks = k
    ranks = torch.arange(n_ranks, device=Um.device)

    U_sel = Um[:, ranks].T.contiguous()  # [n_ranks, m]
    V_sel = Vm[:, ranks].T.contiguous()  # [n_ranks, n]

    # Optionally slice tasks early to avoid unnecessary compute
    if target != -1:
        K = Wts.shape[0]
        if target < 0 or target >= K:
            raise IndexError(f"target={target} out of range for K={K}")
        Wts_use = Wts[target:target + 1]  # [1, m, n]
    else:
        Wts_use = Wts  # [K, m, n]

    if not row_space:
        am_batch = U_sel @ Wm                                   # [n_ranks, n]
        ai_batch = torch.einsum('rm,kmn->rkn', U_sel, Wts_use)   # [n_ranks, K', n]
    else:
        am_batch = V_sel @ Wm.T                                 # [n_ranks, m]
        ai_batch = torch.einsum('rn,kmn->rkm', V_sel, Wts_use)   # [n_ranks, K', m]

    denom = (ai_batch * ai_batch).sum(dim=-1).clamp_min(1e-12)   # [n_ranks, K']
    s = torch.einsum('rd,rkd->rk', am_batch, ai_batch) / denom   # [n_ranks, K']

    alpha_t = torch.tensor(alpha, device=s.device, dtype=s.dtype)

    if target == -1:
        # Original behavior over all tasks
        # pos_all = (s > 0).all(dim=1)        # [n_ranks]
        # mean_s = s.mean(dim=1)              # [n_ranks]
        # eta_vec = torch.where(
        #     pos_all,
        #     torch.maximum(mean_s, alpha_t),
        #     torch.ones_like(mean_s)
        # )
        eta_vec = torch.maximum(s, alpha_t).mean(dim=1)  
    else:
        # Only the target task (now at index 0 after slicing)
        s_t = s[:, 0]                       # [n_ranks]
        pos = s_t > 0
        eta_vec = torch.where(
            pos,
            torch.maximum(s_t, alpha_t),
            torch.ones_like(s_t)
        )

    s_list = eta_vec.to(device=Sm.device, dtype=torch.float32)

    print("s_list: ", s_list.cpu()[:9], s_list.max().item(), s_list.min().item())
    return n_ranks, s_list




def Align(task_vectors, key, Wts, Wm, alpha=1, args=None):
    device = task_vectors[0].vector[key].device
    
    if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key and "embedding" not in key:
        # Wts: [K, m, n], Wm: [m, n]
        Wts = Wts.cuda()
        Wm  = Wm.cuda()
        try:
            Um, Sm, Vh = torch.linalg.svd(Wm, full_matrices=False)  # U:[m,k], S:[k], Vh:[k,n]
        except:
            noise = torch.eye(Wm.shape[0], Wm.shape[1]) * 1e-8
            Um, Sm, Vh = torch.linalg.svd(Wm+noise.cuda(), full_matrices=False)  # U:[m,k], S:[k], Vh:[k,n]
        Vm = Vh.transpose(-2, -1)                               # V:[n,k]
        k = Sm.shape[0]
        Sm_NEW = Sm.clone()

        
        # Parallel computation over multiple ranks R
        n_ranks, s_list = coef_cal_multi_ranks(
            Wts, Wm, Um, Sm_NEW, Vm, k, row_space=args.right_only, alpha=alpha, target=args.target
        )
        
                
        # Singular Value Calibration
        Sm_NEW[:n_ranks] = Sm_NEW[:n_ranks] / s_list
        Wm = (Um @ torch.diag(Sm_NEW) @ Vm.T).to(device)
    
    return Wm


# @profile
def layer_wise_Align(task_vector_avg, task_vectors, args):
    """
    Layer-wise TIES merging function.
    This function merges task vectors layer by layer using TIES method.
    """
    print("Balancing...")
    start_time = time.time()
    args.scaling_coef = 1
    
    with torch.no_grad():
        for key in task_vectors[0].vector:
            print(f"Processing {key}...")
            Wts = torch.stack([tv.vector[key] for tv in task_vectors]) # [T, M, N]
            Wm = task_vector_avg.vector[key]
            
            Wm = Align(task_vectors, key, Wts, Wm, args.alpha, args)
            
            # Reshape back to original layer shape
            task_vector_avg.vector[key] = Wm

    end_time = time.time()
    print(f"Layer-wise Align Time: {end_time - start_time:.1f} seconds")
    return task_vector_avg



