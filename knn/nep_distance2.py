import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager
import math

def nep_worker(knn,P,sid,eid,process_i,res_dict):
    n, k = knn.shape
    pos = np.zeros(n).astype('int32') - 1
    y = np.zeros((k,k)).astype('float32')
    res_dist = []
    for i in tqdm(range(sid,eid)):
        pos[knn[i]] = np.arange(k)
        knn_tmp = knn[knn[i]]
        P_tmp = P[knn[i]]
        y.fill(0)
        x_ind,y_ind = np.where(pos[knn_tmp]>=0)
        pos_index = pos[knn_tmp[x_ind,y_ind]]
        y[x_ind,pos_index] = P_tmp[x_ind,y_ind]
        pos[knn[i]] = -1
        #tmp_dist = np.dot(P[i],y.T)
        tmp_dist = (P[i]+y)*(y!=0)
        tmp_dist = tmp_dist.sum(axis=-1)/2
        res_dist.append(tmp_dist)
    # return dist
    result = np.array(res_dist)
    res_dict[process_i] = result

def nep_distance_opt_mp(knn, dist):
    print("generate nep distance ...")
    processNum = 64 #10
    sigma = 0.5 
    n, k = knn.shape
    P = np.exp(- dist / sigma)   
    ss = P.sum(axis = 1)
    for i in tqdm(range(len(P))):
        P[i] = P[i] / ss[i]
    #P = np.sqrt(P)
    step = math.ceil(n / processNum)
    pool = []
    res_dict = Manager().dict()
    for process_i in range(processNum):
        sid = process_i*step
        eid = min((process_i+1)*step, n)
        t = Process(target=nep_worker,args=(knn,P,sid,eid,process_i,res_dict))
        pool.append(t)
    for process_i in range(processNum):
        pool[process_i].start()
    for process_i in range(processNum):
        pool[process_i].join()
    
    dist = 1 - np.concatenate([res_dict[i] for i in range(processNum)],0)
    #dist = np.concatenate([res_dict[i] for i in range(processNum)],0)
    print('done!!!')
    return dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'knn')
    parser.add_argument('--input_data_path',  metavar = 'IP', type = str, default = "../data/knns/{}/face.npy", help = 'input data path')
    parser.add_argument('--part',  metavar = 'PT', type = str, default = "part1_test", help = 'part name')
    args = parser.parse_args()

    input_data_path = args.input_data_path.format(args.part) 
    output_nbrs_path = input_data_path.replace("face.npy", "knn_nbrs.npz")
    output_dists_path = input_data_path.replace("face.npy", "knn_dists.npz")

    nbrs = np.load(output_nbrs_path)["data"]
    dists = np.load(output_dists_path)["data"]
    dists = np.clip(dists, 0.0, 1.0)
    # l2^2 = (1-ip)*2
    dists = 2 - 2 * dists
    dists = nep_distance_opt_mp(nbrs, dists)
    print(dists)
    
    output_dists_path = input_data_path.replace("face.npy", "knn_dists_trans2.npz")
    np.savez_compressed(output_dists_path, data=dists)
