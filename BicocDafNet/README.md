BicOC-DAFNet
===============


## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/eurecom-asp/RawGAT-ST-antispoofing.git
$ conda create --name RawGAT_ST_anti_spoofing python=3.8.8
$ conda activate RawGAT_ST_anti_spoofing
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are done in the logical access (LA) partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Training
To train the model run:
```
python main.py --track=logical --loss=ce   --lr=0.0001 --batch_size=10
nohup  python main.py --track=logical --loss=ocsoftmax   --lr=0.0001 --batch_size=10    >out_ocAFF.txt &
```

### Testing

To evaluate your own model on LA evaluation dataset:

```
nohup python main.py --track=logical --loss=ocsoftmax --is_eval --eval --model_path='/public/home/acal2okrm7997/g813_u1/zhaobo/RawGAT-ST/models/model_logical_ocsoftmax_100_10_0.0001/epoch_93.pth' --eval_output='eval_scores_ocAFF93.txt' &
```

We also provide a pre-trained models. To use it you can run: 
```
nohup python main.py --track=logical --loss=WCE --is_eval --eval --model_path='Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth' --eval_output='RawGAT_ST_mul_LA_eval_CM_scores.txt' 
```

If you would like to compute scores on development dataset simply run:

```
python main.py --track=logical --loss=WCE --eval --model_path='/path/to/your/best_model.pth' --eval_output='dev_CM_scores_file.txt'
```
Compute the min t-DCF and EER(%) on development dataset
```
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py  dev  'dev_CM_scores_file.txt'
``` 

Compute the min t-DCF and EER(%) on evaluation dataset
```
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py  Eval  '/public/home/acal2okrm7997/g813_u1/zhaobo/RawGAT-ST/eval_scores_ocAFF93.txt'
```






