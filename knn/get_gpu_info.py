import GPUtil
import faiss

def get_gpu_info():
    Gpus = GPUtil.getGPUs()
    gpulist = []
    GPUtil.showUtilization()

    # get gpus
    for gpu in Gpus:
        print('gpu.id:', gpu.id)
        print('GPU total:', gpu.memoryTotal)
        print('GPU usage:', gpu.memoryUsed)
        print('GPU rate:', gpu.memoryUtil * 100)
        # add gpu info 
        gpulist.append([ gpu.id, gpu.memoryTotal, gpu.memoryUsed,gpu.memoryUtil * 100])

    return gpulist

def get_gpu_num():
    try:
        ngpu = faiss.get_num_gpus()
    except:
        ngpu = len(GPUtil.getGPUs())
    return ngpu

def find_available_gpu():
    Gpus = GPUtil.getGPUs()
    memoryUtil = 100
    index = 0
    for gpu in Gpus:
        if gpu.memoryUtil*100 < memoryUtil:
            memoryUtil = gpu.memoryUtil
            index = gpu.id
    return index


