import faiss
faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        
def faiss_search_approx_knn_cpu(query, target, k):
    index = faiss.IndexFlatIP(query.shape[1])
    index.add(target)
    D, I = index.search(query, k)
    return D, I

