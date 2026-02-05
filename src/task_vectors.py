import pickle
import torch
from src.ties_merging_utils import state_dict_to_vector, topk_values_mask, vector_to_state_dict


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, 
                 vector=None, cut_add=False, decompose=False, zero_init=False, tasks20=False):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.cut_add = cut_add
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + finetuned_checkpoint)
                print(pretrained_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint, weights_only=False).state_dict()
                if tasks20:
                    finetuned_state_dict = torch.load(finetuned_checkpoint, weights_only=False)
                else:
                    try:
                        finetuned_state_dict = torch.load(finetuned_checkpoint, weights_only=False).cpu().state_dict()
                    except RuntimeError:
                        finetuned_state_dict = pickle.load(open(finetuned_checkpoint, 'rb')).cpu().state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    
                    if zero_init:
                        self.vector[key] = torch.zeros_like(finetuned_state_dict[key])
                    else:
                        self.vector[key] = finetuned_state_dict[key].cpu() - pretrained_state_dict[key]
                    if decompose and len(self.vector[key].shape) == 2: # SVD decomposition
                        K = max(int(min(self.vector[key].shape) // 10), 1)
                        U, S, V = torch.svd(self.vector[key])
                        self.vector[key] = U[:,:K] @ torch.diag(S[:K]) @ V[:,:K].t()

    
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                if self.vector[key] is None or other.vector[key] is None:
                    new_vector[key] = None
                else:
                    new_vector[key] = self.vector[key] + other.vector[key]
                if self.cut_add:                
                    out = self.vector_revise(self.vector[key], other.vector[key])
                    # out = None
                    if out is not None:
                        new_vector[key] = out
                    
        return TaskVector(vector=new_vector)
    
    
    def to_cuda(self):
        """Move the task vector to GPU."""
        with torch.no_grad():
            for key in self.vector:
                if self.vector[key] is not None:
                    self.vector[key] = self.vector[key].cuda()
                else:
                    print(f'Warning: key {key} is None, not moving to GPU')
        return self
    
    def to_cpu(self):
        """Move the task vector to CPU."""
        with torch.no_grad():
            for key in self.vector:
                if self.vector[key] is not None:
                    self.vector[key] = self.vector[key].cpu()
                else:
                    print(f'Warning: key {key} is None, not moving to CPU')
        return self
    
    def single_revise(self, g_i, g_j):
        _g_i = g_i - torch.dot(g_i, g_j) / torch.dot(g_j, g_j) * g_j
        _g_j = g_j - torch.dot(g_j, g_i) / torch.dot(g_i, g_i) * g_i
        
        return _g_i + _g_j
    
    
    def vector_revise(self, g_i, g_j):
        dim = g_i.dim()
        if dim == 0:
            if torch.sign(g_i) != torch.sign(g_j):
                return g_i*2 if g_i.abs() > g_j.abs() else g_j*2
            else:
                return None
        elif dim == 1:
            cosine_angle = torch.dot(g_i, g_j)
            if cosine_angle < 0:
                return self.single_revise(g_i, g_j)
            else:
                return None
        elif dim == 2:
            out = []
            for row in range(g_j.shape[0]):
                cosine_angle = torch.dot(g_i[row], g_j[row])
                if cosine_angle < 0:
                    out.append(self.single_revise(g_i[row], g_j[row]))
                else:
                    out.append(g_i[row]+g_j[row])
            return torch.stack(out)
        elif dim == 4: # 768, 3, 32, 32
            out = []
            for row in range(g_j.shape[0]):
                cosine_angle = torch.dot(g_i[row].reshape(-1), g_j[row].reshape(-1))
                if cosine_angle < 0:
                    out.append(self.single_revise(g_i[row].reshape(-1), g_j[row].reshape(-1)).reshape(3, 32, 32))
                else:
                    out.append(g_i[row]+g_j[row])
            return torch.stack(out)
        else: 
            return None


    def __sub__(self, other):
        """Subtract two task vectors."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] - other.vector[key]
        return TaskVector(vector=new_vector)
    
    def __mul__(self, other):
        with torch.no_grad():
            new_vector = self.vector.copy()
            for key in self.vector:
                if new_vector[key] is not None:
                    new_vector[key] = self.vector[key] * other
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    @staticmethod
    def weightmerging(self, taskvectors, coefficients, elect_sign=False):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, concat=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
            ####################################################################
            if concat:
                for idx in range(12):
                    c_fc_weight, c_fc_bias = pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_fc.weight, pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_fc.bias
                    linear = torch.nn.Linear(768, 3072*2, bias=True)
                    linear.weight.data = torch.concat([c_fc_weight, c_fc_weight], dim=0)
                    linear.bias.data = torch.concat([c_fc_bias, c_fc_bias], dim=0)
                    pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_fc = linear
                    
                    c_proj_weight, c_proj_bias = pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_proj.weight, pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_proj.bias
                    linear = torch.nn.Linear(3072*2, 768, bias=True)
                    linear.weight.data = torch.concat([c_proj_weight, c_proj_weight], dim=1)
                    linear.bias.data = c_proj_bias
                    pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_proj = linear
            #################################################################### 
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                if concat and 'c_proj.weight' in key:
                    device = self.vector[key].device
                    new_state_dict[key] = (pretrained_state_dict[key].to(device) + scaling_coef * self.vector[key]) / 2
                else:
                    device = self.vector[key].device
                    new_state_dict[key] = pretrained_state_dict[key].to(device) + scaling_coef * self.vector[key]
        
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    
    
    def _apply_to(self, pretrained_checkpoint, scaling_coef=1.0, concat=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
            ####################################################################
            if concat:
                for idx in range(12):
                    c_fc_weight, c_fc_bias = pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_fc.weight, pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_fc.bias
                    linear = torch.nn.Linear(768, 3072*2, bias=True)
                    linear.weight.data = torch.concat([c_fc_weight, c_fc_weight], dim=0)
                    linear.bias.data = torch.concat([c_fc_bias, c_fc_bias], dim=0)
                    pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_fc = linear
                    
                    c_proj_weight, c_proj_bias = pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_proj.weight, pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_proj.bias
                    linear = torch.nn.Linear(3072*2, 768, bias=True)
                    linear.weight.data = torch.concat([c_proj_weight, c_proj_weight], dim=1)
                    linear.bias.data = c_proj_bias
                    pretrained_model.model.visual.transformer.resblocks[idx].mlp.c_proj = linear
            #################################################################### 
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                if concat and 'c_proj.weight' in key:
                    device = self.vector[key].device
                    new_state_dict[key] = (scaling_coef * pretrained_state_dict[key].to(device) + self.vector[key]) / 2
                else:
                    device = self.vector[key].device
                    new_state_dict[key] = scaling_coef * pretrained_state_dict[key].to(device) + self.vector[key]
        
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    
    def trim(self, reset_thresh=30, remove_keys=[], reverse=False):
        flat_tv = state_dict_to_vector(self.vector, remove_keys)

        updated_checks, *_ = topk_values_mask(
            flat_tv, K=reset_thresh, return_mask=False, reverse=reverse
        )    
        
        self.vector = vector_to_state_dict(updated_checks, self.vector, remove_keys)
        
    def scaling(self, ratio, remove_keys=[]):
        flat_tv = state_dict_to_vector(self.vector, remove_keys)
        updated_checks = flat_tv * ratio
        self.vector = vector_to_state_dict(updated_checks, self.vector, remove_keys)
        
    def scale_statistic(self, name, target_keys=[]):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for target_key in target_keys:
            if target_key not in self.vector.keys():
                print(f'Warning: key {target_key} is not present in the task vector')
                continue
            W_flat = self.vector[target_key].cpu().flatten()
            
            # plt.clf()
            # sns.lineplot(data=W_flat)
            # plt.title(f"Task Vecotr of {name} {target_key}")
            # if "mlp" in target_key:
            #     plt.savefig(f"figs/mlp_{name}.png")  
            # else:
            #     plt.savefig(f"figs/attn_{name}.png")      
        
        W_flat = W_flat / W_flat.norm()
        return W_flat

    def clone(self):
        return TaskVector(vector=self.vector.copy())