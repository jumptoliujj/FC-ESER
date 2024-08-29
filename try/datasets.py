import numpy as np
from scipy import sparse
from utils.utils import (read_meta, read_probs, l2norm, knns2ordered_nbrs,
                   intdict2ndarray, Timer)
from tqdm import tqdm


def read_ms1m(feat_path, label_path, knn_graph_path, feature_dim):
    with Timer('read meta and feature'):
        _, idx2lb = read_meta(label_path, verbose=False)
        inst_num = len(idx2lb)
        labels = intdict2ndarray(idx2lb)

        features = read_probs(feat_path, inst_num, feature_dim)
        features = l2norm(features)

    with Timer('read knn graph'):
        knn_graph = np.load(knn_graph_path)['data']
        dists = np.load(knn_graph_path.replace("_nbrs", "_dists_trans2"))['data']
        #dists = np.load(knn_graph_path.replace("_nbrs", "_dists"))['data']
        dists = np.clip(dists, 0.0, 1.0)
        sims = 1 - dists
        #sims = dists
        #dists = (1-sims)*2
        
        dists_ip = np.load(knn_graph_path.replace("_nbrs", "_dists"))['data']
        dists_ip = np.clip(dists_ip, 0.0, 1.0)
        
    #print(dists, knn_graph, labels)
    print(features.shape, labels.shape, dists.shape, knn_graph.shape, dists_ip.shape)

    return features, labels, knn_graph, dists, sims, dists_ip


class PCENetDataset(object):
    def __init__(self, dataset_name, feat_path, label_path, knn_graph_path, feature_dim, k, vaa, vbb, vis):
        self.dataset_name = dataset_name
        self.feature_dim = feature_dim

        data = read_ms1m(feat_path=feat_path, label_path=label_path,
                         knn_graph_path=knn_graph_path, feature_dim=self.feature_dim)

        self.features, self.labels, self.knn_graph, self.dists, self.sims, self.dists_ip = data

        self.k = k

        # --------------------------------------------------
        self.knn_graph = self.knn_graph[:, :k]
        self.sims = self.sims[:, :k]
        self.dists = self.dists[:, :k]
        # --------------------------------------------------
        
        # app
        try_path = "temp/{}_t_{:.2f}_{:.2f}.npz".format(dataset_name ,vaa, vbb)
        vis.slogger.info("try path {}".format(try_path))
        try:
            self.app = np.load(try_path)["data"]
            vis.slogger.info("vaa {} vbb {} knn {} app {}".format(vaa, vbb, self.knn_graph.shape, len(self.app)))
        except:
            a = 0
            b = 0
            self.app = []
            for i in tqdm(range(self.knn_graph.shape[0])):
                flag = 0
                for j in range(1,self.knn_graph.shape[1]):
                    src = i
                    dst = self.knn_graph[src][j]
                    d = self.sims[src][j]

                    if d < vaa: # early stopping positions
                        a += 1
                        flag = 1

                    if flag == 1:
                        if d >= vbb: # edge recall pairs
                            self.app.append([src,dst])
                            b += (self.labels[src]==self.labels[dst])

            vis.slogger.info("vaa {} vbb {} knn {} app {} gt {}".format(vaa, vbb, self.knn_graph.shape, len(self.app), b))
            self.app = np.array(self.app)
            np.savez_compressed(try_path, data=self.app)

    def __getitem__(self, index):

        pair = self.app[index]
        
        f1 = self.features[self.knn_graph[pair[0]]]
        f2 = self.features[self.knn_graph[pair[1]]]

        features = np.concatenate([f1, f2], axis=0)
        adj = features @ features.T
        label = self.labels[pair[0]]==self.labels[pair[1]]

        adj2 = adj
                
        return (features, adj, adj2), label, pair

    def __len__(self):
        return len(self.app)

    def len_nodes(self):
        return self.features.shape[0]

    def get_labels(self):
        return self.labels
