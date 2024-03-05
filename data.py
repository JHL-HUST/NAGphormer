import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
https://github.com/THUDM/GRAND-plus/blob/main/utils/make_dataset.py
"""
def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

"""
https://github.com/THUDM/GRAND-plus/blob/main/utils/make_dataset.py
"""
def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def get_dataset(dataset, pe_dim, split_seed=0):
    file_dir = 'dataset'
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:
        file_path = file_dir + dataset+".pt"

        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        elif dataset == "computer":
            graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        graph = dgl.to_bidirected(graph)

        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
     
        features = torch.cat((features, lpe), dim=1)



    elif dataset in {"aminer", "reddit", "Amazon2M"}:
        file_path = file_dir + dataset + '.pt'
        if os.path.exists(file_path):
            data_list = torch.load(file_path)

            #adj, features, labels, idx_train, idx_val, idx_test

            adj = data_list[0]
            #print(type(adj))
            features = data_list[1]
            labels = data_list[2]
            idx_train = data_list[3]
            idx_val = data_list[4]
            idx_test = data_list[5]
        else:
            import pickle as pkl
            if dataset == 'aminer':
            
                adj = pkl.load(open(os.path.join(file_dir, "{}.adj.sp.pkl".format(dataset)), "rb"))
                features = pkl.load(
                    open(os.path.join(file_dir, "{}.features.pkl".format(dataset)), "rb"))
                labels = pkl.load(
                    open(os.path.join(file_dir, "{}.labels.pkl".format(dataset)), "rb"))
                random_state = np.random.RandomState(split_seed)
                idx_train, idx_val, idx_test = get_train_val_test_split(
                    random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
                idx_unlabel = np.concatenate((idx_val, idx_test))
                features = col_normalize(features)
            elif dataset in ['reddit']:
                adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
                features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
                labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset))) 
                print(labels.shape, list(np.sum(labels, axis=0)))
                random_state = np.random.RandomState(split_seed)
                idx_train, idx_val, idx_test = get_train_val_test_split(
                    random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
                idx_unlabel = np.concatenate((idx_val, idx_test))
                print(dataset, features.shape)
            
            elif dataset in ['Amazon2M']:
                adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
                features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
                labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset)))
                print(labels.shape, list(np.sum(labels, axis=0)))
                random_state = np.random.RandomState(split_seed)
                class_num = labels.shape[1]
                idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20*class_num, val_size=30 * class_num)
                idx_unlabel = np.concatenate((idx_val, idx_test))

            adj = adj + sp.eye(adj.shape[0])
            D1 = np.array(adj.sum(axis=1))**(-0.5)
            D2 = np.array(adj.sum(axis=0))**(-0.5)
            D1 = sp.diags(D1[:, 0], format='csr')
            D2 = sp.diags(D2[0, :], format='csr')

            A = adj.dot(D1)
            A = D2.dot(A)
            adj = A

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)

        graph = dgl.from_scipy(adj)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
    
        features = torch.cat((features, lpe), dim=1)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

        labels = torch.argmax(labels, -1)
        

    return adj, features, labels, idx_train, idx_val, idx_test




