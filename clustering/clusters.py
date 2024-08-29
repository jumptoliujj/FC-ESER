import argparse
import pickle
import infomap
import numpy as np
from tqdm import tqdm
from utils import Timer
from evaluation import evaluate, accuracy


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def get_links(single, links, nbrs, dists, tt, args):
    min_sim = args.theta
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        flag = 0
        # sort - do not sort and use early stopping
        #ind = np.argsort(dists[i])
        #nbrs[i] = nbrs[i][ind]
        #dists[i] = dists[i][ind]
        for j in range(0, len(nbrs[i])):
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                sim = float(1 - dists[i][j])
                if flag == 0:
                    links[(i, nbrs[i][j])] = sim
                    count += 1
            else:
                flag = 1
            
            if flag == 1:
                if args.er:
                    # get edge reacll results into clustering
                    if  dists[i][j] <= (1 - args.delta):
                        if tt[(i,nbrs[i][j])] >= args.eta:
                            links[(i, nbrs[i][j])] = tt[(i,nbrs[i][j])]
                            count += 1
                else:
                    # early stopping
                    break
        # single node
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(nbrs, dists, lbs, tt, args):
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, tt=tt, args=args)

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # infomap run
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # clustering results
    for node in infomapWrapper.iterTree():
        idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
            # print(k, v[2:])
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # print(k, v[1:])

    # single node
    print("single node: {}".format(len(single)))

    keys_len = len(list(label2idx.keys()))

    # merge single node into results
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1

    print("total clusters: {}".format(keys_len))

    idx_len = len(list(idx2label.keys()))
    print("total nodes: {}".format(idx_len))

    # run metrics
    pred_labels = intdict2ndarray(idx2label)
    metrics = ['pairwise', 'bcubed', 'nmi']
    for metric in metrics:
        evaluate(lbs, pred_labels, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'clustering')
    parser.add_argument('--input_data_path',  metavar = 'IP', type = str, default = "../data/knns/{}/face.npy", help = 'input data path')
    parser.add_argument('--part',  metavar = 'PT', type = str, default = "part1_test", help = 'part name')
    parser.add_argument('--er', action = 'store_true', help = 'additional edge recall' )
    parser.add_argument('--theta',  metavar = 'Ptheta', type = float, default = 0.22, help = 'parameters theta')
    parser.add_argument('--delta',  metavar = 'Pdelta', type = float, default = 0.12, help = 'parameters delta')
    parser.add_argument('--eta',  metavar = 'Peta', type = float, default = 0.60, help = 'parameters eta')
    args = parser.parse_args()

    with Timer('All face cluster step'):
        # read knn and nep
        knn_data_path = args.input_data_path.format(args.part)
        nep_dists_path = knn_data_path.replace("face.npy", "knn_dists_trans2.npz")
        nbrs_path = knn_data_path.replace("face.npy", "knn_nbrs.npz")
        
        dists = np.load(nep_dists_path)['data']
        dists = np.clip(dists, 0.0, 1.0)
        nbrs = np.load(nbrs_path)['data']
        print("dists max {} min {}".format(np.max(dists), np.min(dists)))
        print(dists.shape, nbrs.shape)
        
        # read gt labels for metrics
        label_path = knn_data_path.replace("knns", "labels").replace("/face.npy", ".meta")
                   
        true_lb2idxs, true_idx2lb = read_meta(label_path)
        lbs = intdict2ndarray(true_idx2lb)
        
        # additional edge reacll
        if args.er: 
            with open("../er_data/try_{}.json".format("temp5_{}".format(args.part)), 'rb') as f:
                tt = pickle.load(f)
            print("paramtes for fc-eser theta {} delta {} eta {}".format(args.theta, args.delta, args.eta))
        else:
            tt = {}
            print("paramtes for fc-es theta {}".format(args.theta))

        # clustering
        cluster_by_infomap(nbrs, dists, lbs, tt, args)
