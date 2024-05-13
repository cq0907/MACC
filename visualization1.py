from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader_vis import SYSUData, RegDBData, TestData
from data_manager import *
from model_mine import embed_net
from utils import *
import imageio

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=100, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log5/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=2.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=6, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--label_smooth', default='on', type=str, help='performing label smooth or not')
parser.add_argument('--p', default=10, type=int, help='performing label smooth or not')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../data/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '../data/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

transform_train = transforms.Compose(transform_train_list)

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
print('==> Building model..')
net = embed_net(n_class, no_local='off', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
net.to(device)

# model_path = checkpoint_path + 'sysu_c_tri_pcb_on_w_tri_1.0_s6_f256_share_net2_base_gm10_k10_p8_lr_0.1_seed_0_best.t'
model_path = checkpoint_path + 'sysu_c_tri_pcb_on_w_tri_2.0_s6_f256_share_net2_base_gm10_k8_p6_lr_0.1_seed_0_best.t'
if os.path.isfile(model_path):
    print('==> loading checkpoint {}'.format(args.resume))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
    print('==> loaded checkpoint {} (epoch {})'
          .format(args.resume, checkpoint['epoch']))
else:
    print('==> no checkpoint found at {}'.format(args.resume))


def partition(lst, n):
    """
    python partition list
    :param lst: list
    :param n: partitionSize
    :return:
    """
    division = len(lst) / float(n)
    return [list(lst)[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def chunk(lst, size):
    """
    python chunk list
    :param lst: list
    :param size: listSize
    :return:
    """
    return [list(lst)[int(round(size * i)): int(round(size * (i + 1)))] for i in range(int(len(lst) / float(size)) + 1)]

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sampler = IdentityPairSampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size, 0)
trainset.aIndex = sampler.index1  # color index
trainset.pIndex = sampler.index2  # thermal index
trainset.nIndex = sampler.index3  # thermal index

loader_batch = args.batch_size * args.num_pos
trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, num_workers=0, drop_last=True)

net.eval()
for batch_idx, (img1, img2, img3, label1, label2, label3) in enumerate(trainloader):
    img1_ = Variable(img1.cuda())
    img2_ = Variable(img2.cuda())
    img3_ = Variable(img3.cuda())

    out1 = net(img1_, img1_, modal=1)
    out2 = net(img2_, img2_, modal=2)
    out3 = net(img3_, img3_, modal=2)

    out = torch.cat([out1, out2, out3], dim=0).detach().cpu().numpy()
    label = torch.cat([label1, label2, label3], dim=0).numpy()
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(out)
    X_pca = PCA(n_components=2).fit_transform(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, label="PCA")
    plt.legend()
    plt.savefig('ccc.png', dpi=120)
    # plt.show()

    embed()


