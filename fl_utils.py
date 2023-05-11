import torch
import copy

preferred_device = 'cpu'


def compute_cos_coeffs(sigma, global_weights, local_weights):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    coeffs = {}
    for idx, w_local in enumerate(local_weights):
        if idx == 0:
            for k in global_weights.keys():
                if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                coeffs[k] = torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k])))
        else:
            for k in global_weights.keys():
                if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                coeffs[k] = torch.add(coeffs[k], torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k]))))
    return coeffs


def aggregate_cos(sigma, global_weights, local_weights):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    coeffs = compute_cos_coeffs(sigma=sigma, global_weights=global_weights, local_weights=local_weights)
    w_glob = None
    for idx, w_local in enumerate(local_weights):
        if idx == 0:
            w_glob = copy.deepcopy(w_local)
            for k in global_weights.keys():
                if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    w_glob[k] = w_local[k]
                else:
                    beta = torch.div(torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k]))), coeffs[k])
                    w_glob[k] = torch.mul(w_glob[k], beta.to(preferred_device))
        else:
            for k in global_weights.keys():
                if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    w_glob[k] = torch.add(w_glob[k], w_local[k])
                else:
                    beta = torch.div(torch.exp(torch.mul(-sigma, cos(global_weights[k], w_local[k]))), coeffs[k])

                    w_glob[k] = torch.add(w_glob[k], torch.mul(w_local[k], beta.to(preferred_device)))  # FedMobility

    for k in w_glob.keys():
        if 'bn' in k or 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            w_glob[k] = torch.div(w_glob[k], len(local_weights))

    return w_glob


def aggregate_avg(local_weights):
    w_glob = None
    for idx, w_local in enumerate(local_weights):
        if idx == 0:
            w_glob = copy.deepcopy(w_local)
        else:
            for k in w_glob.keys():
                w_glob[k] = torch.add(w_glob[k], w_local[k])

    for k in w_glob.keys():
        w_glob[k] = torch.div(w_glob[k], len(local_weights))

    return w_glob
