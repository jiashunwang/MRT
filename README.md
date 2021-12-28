# MRT

This is an implementation of the NeurIPS'21 paper "Multi-Person 3D Motion Prediction with Multi-Range Transformers".

Please check our [paper](https://arxiv.org/pdf/2111.12073.pdf) and the [project webpage](https://jiashunwang.github.io/MRT/) for more details. 

We will also provide the code to fit our skeleton representation data to [SMPL](https://smpl.is.tue.mpg.de/) data.

## Citation

If you find our code or paper useful, please consider citing:
```
@article{wang2021multi,
  title={Multi-Person 3D Motion Prediction with Multi-Range Transformers},
  author={Wang, Jiashun and Xu, Huazhe and Narasimhan, Medhini and Wang, Xiaolong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Dependencies

Requirements:
- python3.6
- pytorch==1.1.0
- [torch_dct](https://github.com/zh217/torch-dct)
- [AMCParser](https://github.com/CalciferZh/AMCParser)

## Datasets
We provide the data preprocessing code of [CMU-Mocap](http://mocap.cs.cmu.edu/) and [MuPoTS-3D](http://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) (others are coming soon). 
For CMU-Mocap, the dictionary tree is like
``` 
   mocap
   ├── amc_parser.py
   ├── mix_mocap.py
   ├── preprocess_mocap.py
   ├── vis.py
   └── all_asfamc
       └── subjects
           ├── 01
           ...
```
After dowloading the original data, please try
```
python ./mocap/preprocess_mocap.py
python ./mocap/mix_mocap.py
```
For MuPoTS-3D, the dictionary tree is like
``` 
   mupots3d
   ├── preprocess_mupots.py
   ├── vis.py
   └── data
       ├── TS1
       ...
```
After dowloading the original data, please try
```
python ./mocap/preprocess_mupots.py
```
 
## Training
To train our model, please try
```
python train_mrt.py
```

## Evaluation and visualization
We provide the evaluation and visualization code in `test.py`

## Acknowledgement
This work was supported, in part, by grants from DARPA LwLL, NSF CCF-2112665 (TILOS), NSF 1730158 CI-New: Cognitive Hardware and Software Ecosystem Community Infrastructure (CHASE-CI), NSF ACI-1541349 CC\*DNI Pacific Research Platform, and gifts from Qualcomm, TuSimple and Picsart.
Part of our code is based on [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) and [AMCParser](https://github.com/CalciferZh/AMCParser). Many thanks!

