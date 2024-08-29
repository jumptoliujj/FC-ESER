import os
import argparse
import numpy as np
from faiss_gpu import faiss_search_approx_knn

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

def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=True):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs

def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr

def _l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec

if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser(description = 'knn')
    parser.add_argument('--input_data_path',  metavar = 'IP', type = str, default = "../data/features/{}.bin", help = 'input data path')
    parser.add_argument('--part',  metavar = 'PT', type = str, default = "part1_test", help = 'part name')
    parser.add_argument('--k',  metavar = 'K', type = int, default = 80, help = 'k of knn')
    args = parser.parse_args()

    # gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # read
    feat_path = args.input_data_path.format(args.part)
    label_path = args.input_data_path.replace("features/{}.bin", "labels/{}.meta").format(args.part)
    print("input paths", feat_path, label_path)

    _, idx2lb = read_meta(label_path, verbose=True)
    inst_num = len(idx2lb)
    labels = intdict2ndarray(idx2lb)
    
    feature_dim = 256
    features = read_probs(feat_path, inst_num, feature_dim)
    features = _l2norm(features)

    feats = features.astype('float32')
    print("feats shape {}".format(feats.shape))
    
    # knn
    k = args.k #80
    dists, nbrs = faiss_search_approx_knn(feats, feats, k)
    print("dists shape {} nbrs shape {}".format(dists.shape, nbrs.shape))
    
    output_nbrs_path = args.input_data_path.replace("features/{}.bin", "knns/{}/knn_nbrs.npz").format(args.part)
    output_dists_path = args.input_data_path.replace("features/{}.bin", "knns/{}/knn_dists.npz").format(args.part)
    root_path = "/".join(output_nbrs_path.split("/")[:-1])
    print("output paths", output_nbrs_path, output_dists_path, root_path)
    os.system("mkdir -p {}".format(root_path))

    np.savez_compressed(output_nbrs_path, data=nbrs)
    np.savez_compressed(output_dists_path, data=dists)
