# Visible-Infrared Person Re-Identification via Modality Augmentation and Center Constraints
Pytorch code for "Visible-Infrared Person Re-Identification via Modality Augmentation and Center Constraints".

### Results
Dataset | Rank1  | mAP | mINP
 ---- | ----- | ------  | -----
 RegDB | ~94.86% | ~89.30%  | ~78.09%
 SYSU-MM01  | ~73.17% | ~69.45% | ~55.27%
 

### Usage
Our code extends the pytorch implementation of Cross-Modal-Re-ID-baseline in [Github](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). Please refer to the offical repo for details of data preparation.

### Training
Train a model by
```bash
python train_mine.py --dataset regdb --gpu 0 --pcb on --share_net 2 --trial 1 --log_path log0/ --model_path save_model0/
```
**Parameters**: More parameters can be found in the script and code.

**For SYSU-MM dataset**: batch_size=8, num_pos=10, local_feat_dim=256, num_strips=6, label_smooth=off, (p=3.0 for gm_pool).

###  Test
Test a model by
```bash
python test_mine_pcb --dataset regdb --gpu 0
```

```bash
python test_mine_pcb --dataset sysu --gpu 0
```

```bash
python test_mine_pcb --dataset sysu --mode indoor --gpu 0
```
###  Citation

Please kindly cite the following paper in your publications if it helps your research:
```
@inproceedings{chen2023visible,
  title={Visible-Infrared Person Re-identification via Modality Augmentation and Center Constraints},
  author={Chen, Qiang and Xiao, Guoqiang and Wu, Jiahao},
  booktitle={International Conference on Artificial Neural Networks},
  pages={221--232},
  year={2023}
}
```

