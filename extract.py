from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader0 import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mine import embed_net
from utils import *
from loss import OriTripletLoss, CenterTripletLoss, CrossEntropyLabelSmooth, TripletLoss_WRT
from tensorboardX import SummaryWriter
from re_rank import random_walk, k_reciprocal
import scipy.io
import numpy as np

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='../save_model1/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=100, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log10/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
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

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../data/SYSU-MM01'
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
pool_dim = 1536

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=1)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

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
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb)
net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)

            flip_input = fliplr(input)
            flip_input = Variable(flip_input.cuda())
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat_pool = net(input, input, test_mode[0])
                feat_pool_1 = net(flip_input, flip_input, test_mode[0])

                feature_pool = (feat_pool.detach() + feat_pool_1.detach()) / 2
                fnorm_pool = torch.norm(feature_pool, p=2, dim=1, keepdim=True)
                feat_pool = feature_pool.div(fnorm_pool.expand_as(feature_pool))

                gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            else:
                feat_pool, feat_fc = net(input, input, test_mode[0])
                gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
                gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    if args.pcb == 'on':
        return gall_feat_pool
    else:
        return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            flip_input = Variable(flip_input.cuda())
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat_pool = net(input, input, test_mode[1])
                feat_pool_1 = net(flip_input, flip_input, test_mode[1])

                feature_pool = (feat_pool.detach() + feat_pool_1.detach()) / 2
                fnorm_pool = torch.norm(feature_pool, p=2, dim=1, keepdim=True)
                feat_pool = feature_pool.div(fnorm_pool.expand_as(feature_pool))

                query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            else:
                feat_pool, feat_fc = net(input, input, test_mode[1])
                query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
                query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    if args.pcb == 'on':
        return query_feat_pool
    else:
        return query_feat_pool, query_feat_fc


if dataset == 'sysu':
    if args.pcb == 'on':
        query_feat_pool = extract_query_feat(query_loader)
    else:
        query_feat_pool, query_feat_fc = extract_query_feat(query_loader)

    if args.pcb == 'on':
        gall_feat_pool = extract_gall_feat(gall_loader)
    else:
        gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader)

    result = {'gallery_f':gall_feat_pool,'gallery_label':gall_label,'query_f':query_feat_pool,'query_label':query_label}
    scipy.io.savemat('baseline_tsne.mat',result)


