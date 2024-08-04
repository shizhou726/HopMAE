import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from scipy.sparse import diags, coo_matrix
from torch import optim
from sklearn.preprocessing import StandardScaler


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse coo matrix."""
    indices = torch_sparse.indices().numpy()
    values = torch_sparse.val.numpy()
    shape = torch_sparse.shape
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_mask(shape, rate, device):
    probs = torch.full(shape, rate).to(device)
    mask = torch.bernoulli(probs)
    return mask


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def get_loss_masked_feat(x, x_hat, indices):
    return F.mse_loss(x[indices], x_hat[indices], reduction='mean')


def cs_loss(x, x_hat):
    x = F.normalize(x, p=2, dim=-1)
    x_hat = F.normalize(x_hat, p=2, dim=-1)
    loss = (x * x_hat).sum(dim=-1).mean()
    return 1 - loss.mean()


def get_norm_adj(adj):
    D = adj.sum(dim=1)
    D_sqrt_inv = D.pow(-0.5)
    D_sqrt_inv[D_sqrt_inv == float('inf')] = 0
    D_sqrt_inv = torch.diag(D_sqrt_inv)
    norm_adj = D_sqrt_inv @ adj @ D_sqrt_inv
    return norm_adj


def preprocess(adj, feat, hops):
    adj = adj.to_dense()
    norm_adj = get_norm_adj(adj)

    nodes_features = torch.empty(feat.shape[0], hops + 1, feat.shape[1])
    for j in range(feat.shape[0]):
        nodes_features[j, 0, :] = feat[j]
    for i in range(hops):
        feat = torch.matmul(norm_adj, feat)
        for j in range(feat.shape[0]):
            nodes_features[j, i + 1, :] = feat[j]

    return nodes_features

def get_norm_adj_sparse(adj):
    d = adj.sum(axis=1).A1
    D_sqrt_inv = diags(np.reciprocal(np.sqrt(d)))
    norm_adj = D_sqrt_inv @ adj @ D_sqrt_inv
    norm_adj = norm_adj.tocoo()
    data = norm_adj.data

    data[np.isinf(data)] = 0
    norm_adj = coo_matrix((data, (norm_adj.row, norm_adj.col)), shape=norm_adj.shape)
    return norm_adj


def arxiv_preprocess(adj, feat, hops):
    adj = torch_sparse_tensor_to_sparse_mx(adj)
    norm_adj = get_norm_adj_sparse(adj)
    norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)

    nodes_features = torch.empty(feat.shape[0], hops + 1, feat.shape[1])
    for j in range(feat.shape[0]):
        nodes_features[j, 0, :] = feat[j]
    for i in range(hops):
        feat = torch.matmul(norm_adj, feat)
        for j in range(feat.shape[0]):
            nodes_features[j, i + 1, :] = feat[j]

    return nodes_features


def create_optimizer(opt, model, lr, weight_decay):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")