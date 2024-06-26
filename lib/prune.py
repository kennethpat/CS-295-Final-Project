import math
import random
import time 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import torch.nn as nn

from pdb import set_trace as st 
from .quant import *
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from lib.eval import eval_ppl            

def lexsort(keys, dim=-1):
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx

def maximize_total_value(matrix):
    # linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(matrix, maximize=True) 
    return col_indices


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(args, model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            if args.semi_sparse_acc:
                W = subset[name].mask
                
            else:
                W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
        # dev = model.hf_device_map["model.embed_tokens"]
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
    elif "opt" in args.model:
        layers = model.model.decoder.layers


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, args, module):
            super().__init__()
            self.module = module
            self.model = args.model
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']

            raise ValueError
    layers[0] = Catcher(args, layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module
    # print(inps)
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
   
    model.config.use_cache = use_cache
    if "llama" in args.model:
        position_ids = cache['position_ids']
        return inps, outs, attention_mask, position_ids 
    elif "opt" in args.model:
        return inps, outs, attention_mask

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers
        
    per_outneuron = False

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "magnitude":
                W_metric = torch.abs(W)
            elif args.prune_method == "ri":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)

            subset[name].weight.data[W_mask] = 0
            


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                print(f"layer {i} device {dev}")
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            if "norm" in args.model:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, norm=True)
            else:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_ria(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask= prepare_calibration_input(args, model, dataloader, device)
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                # inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            if args.gptq:
                print('Quantizing ...')
                wrapped_layers[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
            
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "wanda":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "ria":
                W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.a
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                if args.reallocation:
                    """
                        Using Heuristic Channel Reallocation
                    """
                    
                    # Try with directly N:M sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    
                    pre_score = torch.sum(W_metric[W_mask==0].type(torch.float32)).item()
                    print("The total value before resort: ", pre_score)
                    
                    
                    # assign importance score to each columns
                    if args.importance_score == "sum":
                        # sum the total value of each column
                        sorted_idx = torch.sort(torch.sum(W_metric, dim=0))[1]
                    elif args.importance_score == "retained_degree_unstructured":
                        # try unstructured pruning
                        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                        W_mask = (W_metric<=thresh)
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask==0)*W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    elif args.importance_score == "retained_degree_per_outneuron":
                        # try unstructured pruning with per output neuron pruning
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                        W_mask = torch.zeros_like(W_metric)==1
                        W_mask.scatter_(1, indices, True)
                        
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask==0)*W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    
                    # channel reallocation
                    index = torch.zeros_like(sorted_idx)
                    for ii in range(1, prune_m+1):
                        if ii % 2 == 1:
                            index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)]
                        else:
                            index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)].flip(0)
                        # index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)]
                    W_metric_resort = W_metric[:, index].clone()
                    
                    W_strip_value = torch.zeros(W_metric.shape[1]//prune_m).to(device)
                    W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric_resort[:,ii:(ii+prune_m)].float()
                            W_mask_permute.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                            W_metric_strip = W_metric_resort[:, ii:(ii+prune_m)]
                            W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii+prune_m)]==0])
                        
                    after_score = torch.sum(W_strip_value.type(torch.float32)).item()
                    print("The total value after heuristic channel reallocation: ", after_score)
                    
                    if args.lsa:
                        """
                            Using linear sum assignment to finetune the N:M
                        """
                        permutation_device = "cuda:7"
                        if args.fast:
                            print("Use Fast!!")
                            fast_name_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                            if name in fast_name_list:
                                blocks = 4
                            elif "up_proj" in name or "gate_proj" in name:
                                blocks = 8
                            else:
                                blocks = 16
                        else:
                            blocks = 1
                        

                        shape = W_metric.shape[1]//prune_m//blocks
                        rows = torch.arange(shape).to(permutation_device)
                        lsa_columns = torch.arange(prune_m).to(permutation_device)
                        def lsa(W_metric, lsa_column, shape, rows, prune_n, prune_m, device):
                            W_metric = W_metric.to(device)
                            score_matrix = torch.zeros(shape, shape).to(device) # score matrix of LSA
                            num_parallel = 1 # How many parallel computation will be used.
                            
                            
                            for row in range(shape//num_parallel):
                                strip_idx = torch.zeros(num_parallel, shape, prune_m).long().to(device)
                                block_columns = torch.arange(prune_m).to(device)
                                columns_mask = block_columns != lsa_column
                                block_columns = block_columns[columns_mask]
                                
                                strip_idx[:, :, 0] = (rows * prune_m).reshape(1, -1) + lsa_column
                                strip_idx[:, :, 1:] = block_columns.reshape(1, 1, -1) + torch.arange(row*num_parallel, (row+1)*num_parallel).reshape(-1, 1, 1).to(device) * prune_m
                                
                                tmp = W_metric[:, strip_idx].transpose(1, 0).transpose(2, 1)
                                
                                W_mask = torch.zeros_like(tmp).to(device)
                                
                                
                                
                                tmp_index = torch.sort(tmp, dim=-1)[1]
                                W_mask.scatter_(dim=-1, index=tmp_index[:, :, :, :prune_n], value=1)
                    
                                score_matrix[:, row*num_parallel:(row+1)*num_parallel] = torch.sum(torch.sum((tmp*(W_mask==0)), dim=-1), dim=-1).transpose(1, 0)
                            
                            score_matrix = score_matrix.transpose(1, 0)
                            
                            col_indices = torch.LongTensor(maximize_total_value(score_matrix.cpu())).to(device)
                            idx = torch.arange(W_metric.shape[1]).long().to(device)
                            idx[rows* prune_m + lsa_column] = col_indices * prune_m + lsa_column
                            
                            return idx
                        
                        z = 0
                        for lsa_column in lsa_columns:
                            t1 = time.time()
                            for ii in range(blocks):
                                index_tmp = index[ii*len(index)//blocks:(ii+1)*len(index)//blocks]
                                permute_idx = lsa(W_metric[:, index_tmp], lsa_column, shape, rows, prune_n, prune_m, permutation_device)
                                permute_idx = permute_idx.to(index.device)

                                index[ii*len(index)//blocks:(ii+1)*len(index)//blocks] = index_tmp[permute_idx]
                            t2 = time.time()
                            W_metric_permute = W_metric[:, index]
                            
                            W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                            for ii in range(W_metric.shape[1]):
                                if ii % prune_m == 0:
                                    tmp = W_metric_permute[:,ii:(ii+prune_m)].float()
                                    W_mask_permute.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                                    W_metric_strip = W_metric_permute[:, ii:(ii+prune_m)]
                                    W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii+prune_m)]==0])
                            print("The total value after linear sum assignment round {}: {}, running time: {}s".format(z, torch.sum(W_strip_value.type(torch.float32)).item(), round(t2-t1, 2)))
                            
                            z += 1
                        
                        
                    W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    W_mask[:, index] = W_mask_permute
                    
                    if args.semi_sparse_acc and prune_n == 2 and prune_m == 4:
                        subset[name].weight = torch.nn.Parameter(to_sparse_semi_structured((W_mask_permute==0)*W[:, index].half()))
                        subset[name].mask = W_mask_permute==0
                    else:
                        subset[name].weight.data[W_mask] = 0

                        
                else:
                    # Directly N:M
                    W_mask = (torch.zeros_like(W_metric) == 1)
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    
                    if args.semi_sparse_acc:
                        subset[name].weight = torch.nn.Parameter(to_sparse_semi_structured(((W_mask==0)*W)).half(), requires_grad=False)
                        subset[name].mask = W_mask==0
                    else:
                        subset[name].weight.data[W_mask] = 0
            else:
                if args.per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                    
                if args.reconstruction:
                    wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def find_layers(module, layers=[nn.Linear], name=''):
    """ Recursively find the layers of a certain type in a module. """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def prune_magnitude_Monte_Carlo(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    initial_sparsity_ratio = args.sparsity_ratio / 10  # Initial sparsity ratio = final ratio / 10
    final_sparsity_ratio = args.sparsity_ratio

    while True:
        current_sparsity_ratio = (torch.sum(torch.stack([p == 0 for p in model.parameters()], dim=0).float()) / torch.sum(torch.tensor([p.numel() for p in model.parameters()]))).item()
        print(f"Current Sparsity Ratio: {current_sparsity_ratio:.4f}")

        if current_sparsity_ratio >= final_sparsity_ratio:
            print(f"Target Sparsity Ratio {final_sparsity_ratio:.4f} achieved.")
            ppl_test = eval_ppl(model, tokenizer, args.eval_dataset, args.test_bs, device)
            print(f"Wikitext Perplexity: {ppl_test:.4f}")
            break

        for i, layer in enumerate(layers):
            subset = find_layers(layer)

            for name, module in subset.items():
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * (final_sparsity_ratio - current_sparsity_ratio))

                # Monte Carlo sampling
                sampled_indices = random.sample(range(num_weights), num_prune)
                sampled_weights = W.flatten()[sampled_indices]

                # Scoring function
                W_metric = torch.abs(sampled_weights)

                # Pruning
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

def gradient_pruning(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items():
            print(f"pruning layer {i} name {name}")
            W = module.weight.data.clone()
            num_weights = W.numel()
            num_prune = int(num_weights * args.sparsity_ratio)

            # Scoring function
            W_metric = torch.abs(W_grads[name])

            # Pruning
            pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
            W_mask = (W_metric <= pruning_threshold).view(W.shape)
            W_mask = W_mask.to(W.device)
            module.weight.data[W_mask] = 0

def entropy_pruning(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items():
            print(f"pruning layer {i} name {name}")
            W = module.weight.data.clone()
            num_weights = W.numel()
            num_prune = int(num_weights * args.sparsity_ratio)

            # Scoring function
            W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension

            # Pruning
            pruning_threshold = torch.sort(W_metric)[0][num_prune]
            W_mask = torch.zeros_like(W)
            W_mask[:, W_metric <= pruning_threshold] = 1
            W_mask = W_mask.to(W.device)
            module.weight.data[W_mask] = 0

def find_layers(module, layers=[torch.nn.Linear], name=''):
    """ Recursively find the layers of a certain type in a module. """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def prune_magnitude_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), matrix=None):
    layers = model.model.layers
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if matrix in name:
                print(f"pruning layer {i} matrix {matrix} name {name}")
                W = module.weight.data
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                module.weight.data[W_mask] = 0

def gradient_pruning_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items(): # q_proj
            if 'mlp' in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                module.weight.data[W_mask] = 0

def entropy_pruning_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if args.matrix == name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                kth_value, kth_index = torch.sort(W_metric)[0].kthvalue(num_prune)
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= kth_value] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

def prune_magnitude_Monte_Carlo_structured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            print(f"pruning layer {i} name {name}")
            W = module.weight.data.clone()
            num_weights = W.numel()
            num_prune = int(num_weights * args.sparsity_ratio)

            # Monte Carlo sampling
            sampled_indices = random.sample(range(num_weights), num_prune)
            sampled_weights = W.flatten()[sampled_indices]

            # Scoring function
            W_metric = torch.abs(sampled_weights)

            # Pruning
            pruning_threshold = sorted(W_metric, reverse=True)[num_prune - 1]
            W_mask = (W.flatten() < pruning_threshold).view(W.shape)
            W_mask = W_mask.to(W.device)
            module.weight.data[W_mask] = 0

def prune_magnitude_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), matrix="q_proj"):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items(): # q_proj
            if 'mlp' in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                module.weight.data[W_mask] = 0

def entropy_pruning_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), matrix="q_proj"):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if matrix in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                kth_value, kth_index = torch.sort(W_metric)[0].kthvalue(num_prune)
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= kth_value] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

def gradient_pruning_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), matrix="q_proj"):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items(): # q_proj
            if 'mlp' in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                module.weight.data[W_mask] = 0

def flip_matrix(A, B, m1, m2):
    """
    Flip elements in matrix A based on overall flip probabilities m1 and m2,
    and the relative magnitudes in matrix B.

    Parameters:
        A (torch.Tensor): A 0-1 matrix.
        B (torch.Tensor): A matrix of the same size as A with weights.
        m1 (float): Overall flip probability of 1s to 0s.
        m2 (float): Overall flip probability of 0s to 1s.

    Returns:
        torch.Tensor: The modified matrix A after flipping.
    """
    # Ensure A and B are of the same size
    assert A.size() == B.size(), "A and B must have the same size"

    # Calculate the total number of 1s and 0s in A
    num_ones = A.sum().item()
    num_zeros = A.numel() - num_ones

    # Calculate the total number of flips for 1s to 0s and 0s to 1s
    total_flips_1_to_0 = int(num_ones * m1)
    total_flips_0_to_1 = int(num_zeros * m2)

    # Mask for 1s and 0s in A
    ones_mask = A == 1
    zeros_mask = A == 0

    # Get the relative probabilities for flipping 1s to 0s
    B_ones = torch.abs(B[ones_mask])
    B_zeros = torch.max(torch.abs(B[zeros_mask])).item() - torch.abs(B[zeros_mask])
    prob_1_to_0 = B_ones.to(torch.float32) / B_ones.to(torch.float32).sum()
    prob_0_to_1 = B_zeros.to(torch.float32) / B_zeros.to(torch.float32).sum()
    # Get the indices of elements to flip based on the calculated probabilities

    if prob_1_to_0.numel() > 10 * total_flips_1_to_0:
        _, indices = torch.topk(prob_1_to_0, 10 * total_flips_1_to_0)
        ones_indices = indices[torch.multinomial(prob_1_to_0[indices], total_flips_1_to_0, replacement=False)]
    else:
        ones_indices = torch.multinomial(prob_1_to_0, total_flips_1_to_0, replacement=False)
    if prob_0_to_1.numel() > 10 * total_flips_0_to_1:
        _, indices = torch.topk(prob_0_to_1, 10 * total_flips_0_to_1)
        zeros_indices = indices[torch.multinomial(prob_0_to_1[indices], total_flips_0_to_1, replacement=False)]
    else:
        zeros_indices = torch.multinomial(prob_0_to_1, total_flips_0_to_1, replacement=False)
    # Flatten the masks to get the linear indices
    ones_flat_indices = torch.nonzero(ones_mask.flatten(), as_tuple=False).squeeze(1)
    zeros_flat_indices = torch.nonzero(zeros_mask.flatten(), as_tuple=False).squeeze(1)

    # Get the indices in the original matrix to flip
    ones_to_flip = ones_flat_indices[ones_indices]
    zeros_to_flip = zeros_flat_indices[zeros_indices]

    # Flatten the matrix A to use advanced indexing for in-place modification
    A_flat = A.flatten()

    # Perform the flips in-place
    A_flat[ones_to_flip] = 0
    A_flat[zeros_to_flip] = 1

    # Reshape the flat matrix back to the original shape (not strictly necessary for in-place)
    A.view_as(A_flat)
    torch.cuda.empty_cache()

import numpy as np
from lib.eval import eval_ppl, eval_zero_shot
import copy
import gc
@torch.no_grad()
def genetic_prune(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    # Objective function to be optimized
    def objective_function(m, mask):
        local_model = copy.deepcopy(m)
        layers = local_model.model.layers
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            for name, module in subset.items():
                module.weight.data[mask[(i, name)]] = 0
        ret = eval_ppl(local_model, tokenizer, args.eval_dataset, args.test_bs, dev)
        del local_model
        gc.collect()
        torch.cuda.empty_cache()
        return ret

    # Initialize parameters
    population_size = 30
    num_generations = 100
    m2 = 0.001 # mutation rate for 0s
    m1 = (1.0 - args.sparsity_ratio) * m2 / args.sparsity_ratio # mutation rate for 1s
    def getFirstIndividule(m):
        local_model = copy.deepcopy(m)
        args.prune_method='ria'
        prune_ria(args, local_model, tokenizer, dev, prune_n, prune_m)
        sparsity_ratio = check_sparsity(args, local_model)
        print(f"first model sparsity sanity check {sparsity_ratio:.4f}")
        sparsity_ratio = check_sparsity(args, m)


        print(f"first model sparsity sanity check {sparsity_ratio:.4f}")
        sparsity_ratio = check_sparsity(args, m)
        print(f"first model sparsity sanity check {sparsity_ratio:.4f}")
        mask = dict()
        layers = local_model.model.layers
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            for name, module in subset.items():
                mask[(i, name)] = (module.weight.data == 0)
        del local_model
        gc.collect()
        torch.cuda.empty_cache()
        return mask
    firstIndividual = getFirstIndividule(model)
    best_individual = firstIndividual
    best_fitness = objective_function(model, best_individual)
    print(f"Generation {0}, Best Fitness: {best_fitness}")


    # Evolution strategy loop
    for generation in range(num_generations):
        for _ in range(population_size):
            candidate = copy.deepcopy(best_individual)
            for (i, name), mask in candidate.items():
                layers = model.model.layers
                for j, layer in enumerate(layers):
                    if i==j:
                        subset = find_layers(layer)
                        for jname, module in subset.items():
                            if name==jname:
                                flip_matrix(mask, module.weight.data, m1, m2)
            # Evaluate the candidate
            candidate_fitness = objective_function(model, candidate)
            print('new child fitness', candidate_fitness)
            # If the candidate is better, update the best individual
            if candidate_fitness < best_fitness:
                del best_individual

                gc.collect()
                torch.cuda.empty_cache()
                best_individual = candidate
                best_fitness = candidate_fitness
            else:
                del candidate
                gc.collect()
                torch.cuda.empty_cache()

        # Print progress
        print(f"Generation {generation}, Best Fitness: {best_fitness}")
        m2 *= 0.5
        m2 = max(m2, 0.0001)
        m1 = (1.0 - args.sparsity_ratio) * m2 / args.sparsity_ratio # mutation rate for 1s

    # Output the best solution
    print("Best Fitness:", best_fitness)

    def get_result(m, mask):
        layers = m.model.layers
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            for name, module in subset.items():
                module.weight.data[mask[(i, name)]] = 0
    get_result(model, best_individual)

# execution time profiler
'''
for name in subset:
    if "mlp" in name:
        args.sparsity_ratio = 0.5
    else:
        args.sparsity_ratio=0.25
'''

# Forward pass through the model
'''
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        lm_logits = model(inputs).logits
# exit()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
'''
