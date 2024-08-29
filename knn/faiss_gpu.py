import os
import gc
import numpy as np
from tqdm import tqdm
import sys
import faiss

from faiss_cpu import faiss_search_approx_knn_cpu

__all__ = ['faiss_search_approx_knn']

def batch_search(index, query, k, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, k), dtype=np.float32)
    nbrs = np.zeros((n, k), dtype=np.int64)

    for sid in tqdm(range(0, n, bs),
                    desc="faiss searching...",
                    disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], k)
    return dists, nbrs


def faiss_search_approx_knn(query,
                            target,
                            k,
                            logger=None,
                            use_gpu=True,
                            bs=int(1e6),
                            verbose=False):

    from get_gpu_info import get_gpu_num
    ngpu = get_gpu_num()
    
    if use_gpu == False or ngpu == 0:
        return faiss_search_approx_knn_cpu(query, target, k)
    
    print("Get {} gpus".format(ngpu))
    cpu_index = faiss.IndexFlatIP(target.shape[1])
    
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    usage_ratio = query.shape[0]*query.shape[1]*4 / (ngpu*15*1000*1000*1000)
    if usage_ratio >= 0.8:
        print("Feature oversize, GPU usage ratio {}, use Float16".format(usage_ratio))
        co.useFloat16 = True
    else:
        print("GPU usage ratio {}, use Float32".format(usage_ratio))
        co.useFloat16 = False
    co.usePrecomputed = False
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)
    
    try:
        gpu_index.add(target)
    except:
        logger.exception("[load features to gpu]")
        sys.exit()
    dists, nbrs = batch_search(gpu_index, query, k=k, bs=bs, verbose=verbose)
    #dists, nbrs = gpu_index.search(query, k=k)
    del gpu_index
    gc.collect()

    return dists, nbrs
