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
    print("loading calibration data")
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    # with torch.no_grad():
    #     if "llama" in args.model:
    #         inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
    #     elif "opt" in args.model:
    #         inps, outs, attention_mask = prepare_calibration_input(args, model, dataloader, device)

    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
            if "mlp" in name:
                

                if args.gptq:
                    print('Quantizing ...')
                    wrapped_layers[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                    )

                print(f"pruning layer {i} name {name}")
                W = subset[name].weight.data.clone()
                if args.prune_method == "wanda":
                    W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                elif args.prune_method == "ria":
                    W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))) ** args.a
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
                                tmp = W_metric[:, ii:(ii + prune_m)].float()
                                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                        pre_score = torch.sum(W_metric[W_mask == 0].type(torch.float32)).item()
                        print("The total value before resort: ", pre_score)

                        # assign importance score to each columns
                        if args.importance_score == "sum":
                            # sum the total value of each column
                            sorted_idx = torch.sort(torch.sum(W_metric, dim=0))[1]
                        elif args.importance_score == "retained_degree_unstructured":
                            # try unstructured pruning
                            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0] * W.shape[1] * args.sparsity_ratio)].cpu()
                            W_mask = (W_metric <= thresh)
                            keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask == 0) * W_metric, dim=0)]
                            sorted_idx = lexsort(keys)
                        elif args.importance_score == "retained_degree_per_outneuron":
                            # try unstructured pruning with per output neuron pruning
                            sort_res = torch.sort(W_metric, dim=-1, stable=True)
                            indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                            W_mask = torch.zeros_like(W_metric) == 1
                            W_mask.scatter_(1, indices, True)

                            keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask == 0) * W_metric, dim=0)]
                            sorted_idx = lexsort(keys)

                        # channel reallocation
                        index = torch.zeros_like(sorted_idx)
                        for ii in range(1, prune_m + 1):
                            if ii % 2 == 1:
                                index[ii - 1::prune_m] = sorted_idx[int(W_metric.shape[1] * (ii - 1) / prune_m):int(W_metric.shape[1] * ii / prune_m)]
                            else:
                                index[ii - 1::prune_m] = sorted_idx[int(W_metric.shape[1] * (ii - 1) / prune_m):int(W_metric.shape[1] * ii / prune_m)].flip(0)

                        W_metric_resort = W_metric[:, index].clone()
                        W_strip_value = torch.zeros(W_metric.shape[1] // prune_m).to(device)
                        W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric_resort[:, ii:(ii + prune_m)].float()
                                W_mask_permute.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                                W_metric_strip = W_metric_resort[:, ii:(ii + prune_m)]
                                W_strip_value[ii // prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii + prune_m)] == 0])

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

                            shape = W_metric.shape[1] // prune_m // blocks
                            rows = torch.arange(shape).to(permutation_device)
                            lsa_columns = torch.arange(prune_m).to(permutation_device)

                            def lsa(W_metric, lsa_column, shape, rows, prune_n, prune_m, device):
                                W_metric = W_metric.to(device)
                                score_matrix = torch.zeros(shape, shape).to(device)  # score matrix of LSA
                                num_parallel = 1  # How many parallel computation will be used.

                                for row in range(shape // num_parallel):
                                    strip_idx = torch.zeros(num_parallel, shape, prune_m).long().to(device)
                                    block_columns = torch.arange(prune_m).to(device)
                                    columns_mask = block_columns % prune_m == lsa_column
                                    row_indices = row * num_parallel + torch.arange(num_parallel)
                                    block_columns[columns_mask] = -1
                                    strip_idx[:, row_indices, block_columns[~columns_mask]] = block_columns[columns_mask]

                                    rows_idx = torch.ones(prune_n).long().to(device) * row_indices[:, None]
                                    for column in range(prune_m):
                                        tmp_score = torch.sum(W_metric[rows_idx, strip_idx[:, :, column], :], dim=-1)
                                        score_matrix[row_indices, tmp_score] = column

                                print("score matrix completed.")
                                score_matrix = torch.topk(score_matrix, prune_n, dim=-1)[1]
                                for i in range(len(score_matrix)):
                                    assignment_rows = rows[i::len(score_matrix)]
                                    assignment_cols = torch.zeros_like(assignment_rows)
                                    for j in range(len(assignment_rows)):
                                        assignment_cols[j] = score_matrix[i, j]

                                    lsa_column.scatter_(0, assignment_rows, assignment_cols)
                                return lsa_column

                            lsa_columns = lsa(W_metric_resort, lsa_columns, shape, rows, prune_n, prune_m, permutation_device)
                            rows_idx = torch.arange(W_metric.shape[1] // prune_m * prune_n).to(device)
                            W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                            for ii in range(W_metric.shape[1] // prune_m):
                                if ii % prune_m == 0:
                                    W_mask_permute.scatter_(1, ii + torch.topk(W_metric_resort[:, ii:(ii + prune_m)], prune_n, dim=1, largest=False)[1], True)
                                    block_row_idx = rows_idx[rows_idx // prune_n == ii]
                                    block_row_idx = torch.arange(W_metric.shape[0])[block_row_idx]
                                    row_idx = block_row_idx.repeat(prune_m).reshape(W_metric.shape[1] // prune_m, prune_m)
                                    for j in range(len(row_idx)):
                                        row = row_idx[j]
                                        col = lsa_columns[row].to(device)
                                        row += ii
                                        W_mask_permute[row, col] = False

                            W_mask.scatter_(1, index, W_mask_permute)

                        subset[name].weight.data[W_mask] = 0
                        subset[name].scaler_row[W_mask] = 0

                        subset[name].weight.data = W * (1 - W_mask)
                        print("The W_mask sum after heuristic channel reallocation: ", torch.sum(W_mask).item())
                        print("The W_mask sum after heuristic channel reallocation per block: ", torch.sum(W_mask[:, :W_metric.shape[1] // prune_m * prune_m]).item())

                    else:
                        """
                        Using direct N:M sparsity
                        """
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii:(ii + prune_m)].float()
                                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                        subset[name].weight.data[W_mask] = 0
                        subset[name].scaler_row[W_mask] = 0

                        subset[name].weight.data = W * (1 - W_mask)
                        print("The W_mask sum after direct N:M sparsity: ", torch.sum(W_mask).item())

    model.config.use_cache = use_cache


'''
def prune_ria(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibration data")
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
        torch.cuda.empty_cache()

        for name in subset:
            if args.gptq:
                print('Quantizing ...')
                wrapped_layers[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )

            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "wanda":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            elif args.prune_method == "ria":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))) ** args.a
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            if prune_n != 0:
                if args.reallocation:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                    pre_score = torch.sum(W_metric[W_mask == 0].type(torch.float32)).item()
                    print("The total value before resort: ", pre_score)

                    if args.importance_score == "sum":
                        sorted_idx = torch.sort(torch.sum(W_metric, dim=0))[1]
                    elif args.importance_score == "retained_degree_unstructured":
                        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0] * W.shape[1] * args.sparsity_ratio)].cpu()
                        W_mask = (W_metric <= thresh)
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask == 0) * W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    elif args.importance_score == "retained_degree_per_outneuron":
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask = torch.zeros_like(W_metric) == 1
                        W_mask.scatter_(1, indices, True)

                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask == 0) * W_metric, dim=0)]
                        sorted_idx = lexsort(keys)

                    index = torch.zeros_like(sorted_idx)
                    for ii in range(1, prune_m + 1):
                        if ii % 2 == 1:
                            index[ii - 1::prune_m] = sorted_idx[int(W_metric.shape[1] * (ii - 1) / prune_m):int(W_metric.shape[1] * ii / prune_m)]
                        else:
                            index[ii - 1::prune_m] = sorted_idx[int(W_metric.shape[1] * (ii - 1) / prune_m):int(W_metric.shape[1] * ii / prune_m)].flip(0)

                    W_metric_resort = W_metric[:, index].clone()
                    W_strip_value = torch.zeros(W_metric.shape[1] // prune_m).to(device)
                    W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric_resort[:, ii:(ii + prune_m)].float()
                            W_mask_permute.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                            W_metric_strip = W_metric_resort[:, ii:(ii + prune_m)]
                            W_strip_value[ii // prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii + prune_m)] == 0])

                    after_score = torch.sum(W_strip_value.type(torch.float32)).item()
                    print("The total value after heuristic channel reallocation: ", after_score)

                    if args.lsa:
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

                        shape = W_metric.shape[1] // prune_m // blocks
                        rows = torch.arange(shape).to(permutation_device)
                        lsa_columns = torch.arange(prune_m).to(permutation_device)

                        def lsa(W_metric, lsa_column, shape, rows, prune_n, prune_m, device):
                            W_metric = W_metric.to(device)
                            score_matrix = torch.zeros(shape, shape).to(device)

                            for row in range(shape // num_parallel):
                                strip_idx = torch.zeros(num_parallel, shape, prune_m).long().to(device)
                                block_columns = torch.arange(prune_m).to(device)
                                columns_mask = block_columns % prune_m == lsa_column
                                row_indices = row * num_parallel + torch.arange(num_parallel)
                                block_columns[columns_mask] = -1
                                strip_idx[:, row_indices, block_columns[~columns_mask]] = block_columns[columns_mask]

                                rows_idx = torch.ones(prune_n).long().to(device) * row_indices[:, None]
                                for column in range(prune_m):
                                    tmp_score = torch.sum(W_metric[rows_idx, strip_idx[:, :, column], :], dim=-1)
                                    score_matrix[row_indices, tmp_score] = column

                            print("score matrix completed.")
                            score_matrix = torch.topk(score_matrix, prune_n, dim=-1)[1]
                            for i in range(len(score_matrix)):
                                assignment_rows = rows[i::len(score_matrix)]
                                assignment_cols = torch.zeros_like(assignment_rows)
                                for j in range(len(assignment_rows)):
                                    assignment_cols[j] = score_matrix[i, j]

                                lsa_column.scatter_(0, assignment_rows, assignment_cols)
                            return lsa_column

                        lsa_columns = lsa(W_metric_resort, lsa_columns, shape, rows, prune_n, prune_m, permutation_device)
                        rows_idx = torch.arange(W_metric.shape[1] // prune_m * prune_n).to(device)
                        W_mask_permute = (torch.zeros_like(W_metric) == 1)
                        for ii in range(W_metric.shape[1] // prune_m):
                            if ii % prune_m == 0:
                                W_mask_permute.scatter_(1, ii + torch.topk(W_metric_resort[:, ii:(ii + prune_m)], prune_n, dim=1, largest=False)[1], True)
                                block_row_idx = rows_idx[rows_idx // prune_n == ii]
                                block_row_idx = torch.arange(W_metric.shape[0])[block_row_idx]
                                row_idx = block_row_idx.repeat(prune_m).reshape(W_metric.shape[1] // prune_m, prune_m)
                                for j in range(len(row_idx)):
                                    row = row_idx[j]
                                    col = lsa_columns[row].to(device)
                                    row += ii
                                    W_mask_permute[row, col] = False

                        W_mask.scatter_(1, index, W_mask_permute)

                    subset[name].weight.data[W_mask] = 0
                    subset[name].scaler_row[W_mask] = 0

                    subset[name].weight.data = W * (1 - W_mask)
                    print("The W_mask sum after heuristic channel reallocation: ", torch.sum(W_mask).item())
                    print("The W_mask sum after heuristic channel reallocation per block: ", torch.sum(W_mask[:, :W_metric.shape[1] // prune_m * prune_m]).item())

                else:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                    subset[name].weight.data[W_mask] = 0
                    subset[name].scaler_row[W_mask] = 0

                    subset[name].weight.data = W * (1 - W_mask)
                    print("The W_mask sum after direct N:M sparsity: ", torch.sum(W_mask).item())

        del wrapped_layers, subset, W, W_metric, W_mask
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
'''

def find_layers(module, layers=[nn.Linear], name=''):
    """ Recursively find the layers of a certain type in a module. """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

'''
# def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
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
            subset[name].weight.data[W_mask] = 0

# def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    num_iterations = 10  # Number of iterations for iterative pruning
    initial_sparsity_ratio = 0.1  # Initial sparsity ratio
    sparsity_ratio_increment = (args.sparsity_ratio - initial_sparsity_ratio) / (num_iterations - 1)  # Increment in sparsity ratio per iteration

    for iteration in range(num_iterations):
        current_sparsity_ratio = initial_sparsity_ratio + iteration * sparsity_ratio_increment

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            for name in subset:
                print(f"pruning layer {i} name {name}")
                W = subset[name].weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * current_sparsity_ratio)

                # Monte Carlo sampling
                sampled_indices = random.sample(range(num_weights), num_prune)
                sampled_weights = W.flatten()[sampled_indices]

                # Scoring function
                W_metric = torch.abs(sampled_weights)

                # Pruning
                pruning_threshold = sorted(W_metric, reverse=True)[num_prune - 1]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                subset[name].weight.data[W_mask] = 0

        print(f"Iteration {iteration + 1}/{num_iterations}, Sparsity Ratio: {current_sparsity_ratio:.4f}")

# def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    #num_iterations = 10  # Number of iterations for iterative pruning GET RID OF THIS LINE
    initial_sparsity_ratio = args.sparsity_ratio/10  # Initial sparsity ratio = final ratio/10
    sparsity_ratio_increment = (args.sparsity_ratio - initial_sparsity_ratio) / (num_iterations - 1)  # Increment in sparsity ratio per iteration

    for iteration in range(num_iterations):#change this to a while loop
        #call pytorch function count 0s in weights. Size of weights is matrix size. 0/matrix size = current sparsity ratio
        current_sparsity_ratio = initial_sparsity_ratio + iteration * sparsity_ratio_increment
        #break if curr_sparsity > final_sparsity output model's perplexity and curr_sparsity

        for i, layer in enumerate(layers):
            subset = find_layers(layer)

            for name, module in subset.items():
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * current_sparsity_ratio)

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

        print(f"Iteration {iteration + 1}/{num_iterations}, Sparsity Ratio: {current_sparsity_ratio:.4f}")
'''
        
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
                # W = module.weight.data.clone()
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
                # W_mask = W_mask.to(W.device)
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
            # W = module.weight.data.clone()
            num_weights = W.numel()
            num_prune = int(num_weights * args.sparsity_ratio)

            # Scoring function
            W_metric = torch.abs(W_grads[name])

            # Pruning
            pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
            W_mask = (W_metric <= pruning_threshold).view(W.shape)
            # W_mask = W_mask.to(W.device)
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
            # W = module.weight.data.clone()
            num_weights = W.numel()
            num_prune = int(num_weights * args.sparsity_ratio)

            # Scoring function
            W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension

            # Pruning
            pruning_threshold = torch.sort(W_metric)[0][num_prune]
            W_mask = torch.zeros_like(W)
            W_mask[:, W_metric <= pruning_threshold] = 1
            # W_mask = W_mask.to(W.device)
            module.weight.data[W_mask] = 0


def find_layers(module, layers=[torch.nn.Linear], name=''):
    """ Recursively find the layers of a certain type in a module. """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

# def prune_magnitude_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items():
            if "q_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "k_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "v_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "o_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "gate_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "up_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "down_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

def prune_magnitude_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers
    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items(): # q_proj
            if args.matrix == name:
                print(f"pruning layer {i} name {name}")
                # W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                # W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
'''
def gradient_pruning_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items():
            if "q_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "k_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "v_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "o_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "gate_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "up_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "down_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
'''

def gradient_pruning_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items():
            if args.matrix == name:
                print(f"pruning layer {i} name {name}")
                # W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                # W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
'''
# def entropy_pruning_layer(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, module in subset.items():
            if "q_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "k_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "v_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "o_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "gate_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "up_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

            elif "down_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                num_prune = min(num_prune, W_metric.size(0))  # Ensure num_prune is not greater than the size of W_metric along dimension 0
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
'''

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
                # W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                kth_value, kth_index = torch.sort(W_metric)[0].kthvalue(num_prune)
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= kth_value] = 1
                # W_mask = W_mask.to(W.device)
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
            # W_mask = W_mask.to(W.device)
            module.weight.data[W_mask] = 0

'''
# def gradient_pruning_structured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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

# def entropy_pruning_structured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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



# def prune_magnitude_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W.flatten() < pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
'''

def prune_magnitude_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), matrix="q_proj"):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if matrix in name:
                print(f"pruning layer {i} name {name}")
                # W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W)
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                # W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0

'''
# def entropy_pruning_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = -(W * torch.log(W + 1e-8)).sum(dim=0)  # Entropy along output dimension
                pruning_threshold = torch.sort(W_metric)[0][num_prune]
                W_mask = torch.zeros_like(W)
                W_mask[:, W_metric <= pruning_threshold] = 1
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0


# def gradient_pruning_layer_structured(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # if "llama" in args.model:
    layers = model.model.layers
    # elif "opt" in args.model:
    #     layers = model.model.decoder.layers

    # Get weight gradients
    W_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                pruning_threshold = W_metric.kthvalue(num_prune, dim=0, keepdim=True)[0]
                W_mask = (W_metric <= pruning_threshold).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
'''

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
        for name, module in subset.items():
            if matrix in name:
                print(f"pruning layer {i} name {name}")
                W = module.weight.data.clone()
                num_weights = W.numel()
                num_prune = int(num_weights * args.sparsity_ratio)
                W_metric = torch.abs(W_grads[name])
                kth_value, kth_index = W_metric.view(-1).kthvalue(num_prune)
                W_mask = (W_metric <= kth_value).view(W.shape)
                W_mask = W_mask.to(W.device)
                module.weight.data[W_mask] = 0
