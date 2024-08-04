import dgl
import numpy as np
import torch

from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset, WikiCSDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from ogb.nodeproppred import DglNodePropPredDataset
from utils import arxiv_preprocess, preprocess


def get_dataset(dataset_name):
    if dataset_name == 'cora':
        return CoraGraphDataset(verbose=False)
    elif dataset_name == 'pubmed':
        return PubmedGraphDataset(verbose=False)
    elif dataset_name == 'citeseer':
        return CiteseerGraphDataset(verbose=False)
    elif dataset_name == 'wiki-cs':
        return WikiCSDataset(verbose=False)
    elif dataset_name == "photo":
        return AmazonCoBuyPhotoDataset(verbose=False)
    elif dataset_name == "computer":
        return AmazonCoBuyComputerDataset(verbose=False)
    elif dataset_name == "cs":
        return CoauthorCSDataset(verbose=False)
    elif dataset_name == "physics":
        return CoauthorPhysicsDataset(verbose=False)
    elif dataset_name == 'ogbn-arxiv':
        return DglNodePropPredDataset(name='ogbn-arxiv')
    else:
        raise NotImplementedError


def load_dataset(dataset_name):
    dataset = get_dataset(dataset_name)
    
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        graph = dataset[0]
        feat = graph.ndata['feat']
        label = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    elif dataset_name == 'wiki-cs':
        graph = dataset[0]
        feat = graph.ndata['feat']
        label = graph.ndata['label']
        # 20 different splits for the WikiCS dataset
        # train_mask [num_nodes, 20]
        # val_mask [num_nodes, 20]
        # test_mask [num_nodes, 20]
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    elif dataset_name in ['photo', 'computer', 'cs', 'physics']:
        graph = dataset[0]
        feat = graph.ndata['feat']
        label = graph.ndata['label']
        num_nodes = graph.num_nodes()
        idx = np.arange(num_nodes)
        np.random.shuffle(idx)
        perm = torch.tensor(idx)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:int(num_nodes * 0.1)]] = True
        val_mask[perm[int(num_nodes * 0.1):int(num_nodes * 0.2)]] = True
        test_mask[perm[int(num_nodes * 0.2):]] = True
    elif dataset_name == 'ogbn-arxiv':
        graph, label = dataset[0]
        label = label.squeeze()
        feat = graph.ndata['feat']
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    else:
        raise NotImplementedError

    graph = dgl.to_bidirected(graph)
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    adj = graph.adjacency_matrix()

    return {'graph': graph,
            'feat': feat,
            'adj': adj,
            'label': label,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask}


if __name__ == "__main__":
    dataset = 'ogbn-arxiv'
    hops = 15
    data_dict = load_dataset(dataset)
    if dataset == 'ogbn-arxiv':
        feat = arxiv_preprocess(data_dict['adj'], data_dict['feat'], hops)
    else:
        feat = preprocess(data_dict['adj'], data_dict['feat'], hops)
    torch.save(feat, f'./dataset/{dataset}_feat_{hops}_{list(feat.shape)}.pt')
    print('done')
