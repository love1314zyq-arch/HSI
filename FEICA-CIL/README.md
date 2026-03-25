# FEICA-CIL
Code for Feature Space Expansion and Compression with Spatial-spectral Augmentation for Hyperspectral Image Class-Incremental Learning.

## To install the environment:  
Models were trained on the Ubutnu 20.04 system. The GPU used in training is Nvidia RTX 4090. The model is built with pytorch 1.11.0, and torchvision 0.12.0.  
```python
conda env create -f environment.yml
```
## Run model training  
### Salinas dataset 
```python
python3 -mhylearn --options options/LSC/lsc_SalinasA.yaml options/data/SalinasA.yaml --initial-increment 4 --increment 4  --device 0 --label LSC_SalinasA_4steps
```
### University of Pavia dataset 
```python
python3 -mhylearn --options options/LSC/lsc_PAU.yaml options/data/PAU.yaml --initial-increment 3 --increment 3  --device 0 --label LSC_PAU_4steps
```
### WHU-Hi-LongKou dataset 
```python
python3 -mhylearn --options options/LSC/lsc_LK.yaml options/data/longkou.yaml --initial-increment 3 --increment 3  --device 0 --label LSC_LK_4steps
```
### WHU-Hi-HanChuan dataset 
```python
python3 -mhylearn --options options/LSC/lsc_HC.yaml options/data/hanchuan.yaml --initial-increment 4 --increment 4  --device 0 --label LSC_HC_4steps
```
## Results
![FEICA-CIL](./Img/lk_revise.png)  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline"> Figure 1. Classification maps of comparative methods on the Longkou dataset. (a) Bic, (b) PODNet, (c) FORSTER, (d) DRC, (e) FEICA-CIL.</center> 
 <br/>  
 <br/>  
 <br/>  
 <br/>  
 
![FEICA-CIL](./Img/san_revise.png)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline"> Figure 2. Classification maps of comparative methods on the Salinas dataset. (a) Bic, (b) PODNet, (c) FORSTER, (d) DRC, (e) FEICA-CIL.</center>  

## Acknowledgement
The implementation extends and refines work originally developed by
```
@inproceedings{douillard2020ghost,
    title={Insight From the Future for Continual Learning},
    author={Arthur Douillard and Eduardo Valle and Charles Ollion and Thomas Robert and Matthieu Cord},
    booktitle={arXiv preprint library},
    year={2020}
}
```
and 
```
@ARTICLE{NCSC-TGRS-2022,
  author={Cai, Yaoming and Zhang, Zijia and Ghamisi, Pedram and  Ding, Yao and Liu, Xiaobo and Cai, Zhihua and Gloaguen, Richard}
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Superpixel Contracted Neighborhood Contrastive Subspace Clustering Network for Hyperspectral Image}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2022.3179637}}
```
