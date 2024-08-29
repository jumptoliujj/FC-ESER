# Face Clustering via Early Stopping and Edge Recall

This repo provides an official implementation for FC-ESER [1], including unsupervised face clustering and supervised face clustering.  

## Paper
[1] Face Clustering via Early Stopping and Edge Recall, [https://arxiv.org/abs/2408.13431](https://arxiv.org/abs/2408.13431).

## Datasets Structure  
```
data/
├── features
│   ├── part0_train.bin
│   ├── part1_test.bin
│   ├── part3_test.bin
│   └── ...
└── labels
    ├── part0_train.meta
    ├── part1_test.meta
    ├── part3_test.meta
    └── ...
```

Datasets can be obtained from:  
- MS1M: [https://github.com/yl-1993/learn-to-cluster](https://github.com/yl-1993/learn-to-cluster).
- MSMT17: [https://github.com/damo-cv/Ada-NETS](https://github.com/damo-cv/Ada-NETS).
- Veri776: Coming soon.

## Run  
- KNN Graph Construction & Neighbor-based Edge Probability
```
cd knn
## KNN
python knn.py --input_data_path ../data/features/{}.bin --part part1_test --k 80
## NEP, one of the most important steps in FC-ES and FC-ESER
python nep_distance2.py --input_data_path ../data/knns/{}/face.npy --part part1_test
```

- CLustering
```
cd clustering
## Unsupervised FC-ES
python clusters.py --input_data_path ../data/knns/{}/face.npy --part part1_test --theta 0.22
## Supervised FC-ESER
python clusters.py --input_data_path ../data/knns/{}/face.npy --part part1_test --theta 0.22 --er --delta 0.12 --eta 0.60
```

- Edge Recall. Run this step before supervised FC-ESER. Training and testing refer to [LCE-PCENet](https://github.com/illian01/lce-pcenet).
```
cd try
## Training
python main.py
## Testing
python test.py
```

## Results
- MS1M 5.21M Part 

|   Methods   |     FP    |     FB    |
|:-----------:|:---------:|:---------:|
|   [Ada-NETS](https://github.com/damo-cv/Ada-NETS)  |   83.99   |   83.28   |
| [Chen et al.](https://github.com/echoanran/on-mitigating-hard-clusters) |   86.94   |   86.06   |
|   [FaceMap](https://github.com/bd-if/Adapt-InfoMap)   |   86.37   |   86.29   |
| [Wang et al.](https://github.com/thomas-wyh/b-attention) |   85.40   |   86.76   |
|    [LCEPCE](https://github.com/illian01/lce-pcenet)   |   87.35   |   87.28   |
|    FC-ES    |   88.75   |   87.23   |
| **FC-ESER** | **89.40** | **88.80** |

## Acknowledgement
Some codes are based on the publicly available codebase [Learn-to-cluster](https://github.com/yl-1993/learn-to-cluster), [LCE-PCENet](https://github.com/illian01/lce-pcenet), [FaceMap](https://github.com/bd-if/Adapt-InfoMap).

## Citation
```
@misc{FC-ESER,
      title={Face Clustering via Early Stopping and Edge Recall}, 
      author={Junjie Liu},
      year={2024},
      eprint={2408.13431},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13431}, 
}
```