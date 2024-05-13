#!/usr/bin/env bash

python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log6/ --model_path save_model6/
python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log7/ --model_path save_model7/
python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log8/ --model_path save_model8/
python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log9/ --model_path save_model9/

#python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log4/ --model_path save_model4/ --sh 0.4
#python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log5/ --model_path save_model5/ --sh 0.3
#python train_mine1.py --dataset sysu --gpu 3 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log6/ --model_path save_model6/ --sh 0.2

#python train_mine1.py --dataset sysu --gpu 2 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log7/ --model_path save_model7/ --sh 0.4
#python train_mine1.py --dataset sysu --gpu 2 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log8/ --model_path save_model8/ --sh 0.3
#python train_mine1.py --dataset sysu --gpu 2 --pcb on --share_net 2 --w_center 1 --p 3 --label_smooth off --num_pos 10 --log_path log9/ --model_path save_model9/ --sh 0.2



#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 1
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 2
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 3
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 4
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 5
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 6
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 7
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 8
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 9
#python train_mine2.py --dataset regdb --gpu 3 --pcb on --share_net 2 --w_center 2 --trial 10

