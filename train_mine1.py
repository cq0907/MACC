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
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mine1 import embed_net
from utils import *
from loss import pdist_torch, CrossEntropyLabelSmooth, CenterWeightTripletLoss, JSDivLoss, DivLoss, CenterTVTripletLoss
from tensorboardX import SummaryWriter
from re_rank import random_walk, k_reciprocal
from IPython import embed

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
parser.add_argument('--model_path', default='save_model3/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=100, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log3/', type=str,
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
parser.add_argument('--p', default=10, type=int, help='performing label smooth or not')
parser.add_argument('--sh', default=0.4, type=float, help='performing label smooth or not')

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
    # test_mode = [1, 2]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset + '_c_tri_pcb_{}_w_tri_{}'.format(args.pcb, args.w_center)
if args.pcb == 'on':
    suffix = suffix + '_s{}_f{}'.format(args.num_strips, args.local_feat_dim)

suffix = suffix + '_share_net{}'.format(args.share_net)
if args.method == 'agw':
    suffix = suffix + '_agw_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_gm10_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

# vis_log_dir = args.vis_log_path + suffix + '/'
# if not os.path.isdir(vis_log_dir):
#     os.makedirs(vis_log_dir)
# writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

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
    trainset = SYSUData(data_path, transform=transform_train, sh=args.sh)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train, sh=args.sh)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

n_class = len(np.unique(trainset.train_color_label))
gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=n_class, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=n_class, shuffle=False, num_workers=args.workers)

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
    net = embed_net(n_class, no_local='off', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb,
                    local_feat_dim=args.local_feat_dim, num_strips=args.num_strips, p=args.p)
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
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
if args.label_smooth == 'off':
    criterion_id = nn.CrossEntropyLoss()
else:
    criterion_id = CrossEntropyLabelSmooth(n_class)

loader_batch = args.batch_size * args.num_pos
# criterion_tri= CenterTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_tri = CenterWeightTripletLoss()
criterion_id.to(device)
criterion_tri.to(device)

criterion_metric = DivLoss()
criterion_metric.to(device)

criterion_metric1 = JSDivLoss()
criterion_metric1.to(device)

if args.optim == 'sgd':
    if args.pcb == 'on':
        ignored_params = list(map(id, net.local_conv_list.parameters())) \
                         + list(map(id, net.fc_list.parameters())) \
                         + list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.local_conv_list.parameters(), 'lr': args.lr},
            {'params': net.fc_list.parameters(), 'lr': args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

# memory = torch.ones((n_class, 1536)).cuda()
# class data_collector(Metric):
#     def reset(self):
#         self.rgb_feat = []
#         self.ca_feat = []
#         self.ir_feat = []
#         self.ids = []
#     def update(self, output):
#         rgb_feat, ca_feat, ir_feat, ids = output
#         self.rgb_feat.append(rgb_feat)
#         self.ca_feat.append(ca_feat)
#         self.ir_feat.append(ir_feat)
#         self.ids.append(ids)
#     def compute(self):
#         rgb_feat = torch.cat(self.rgb_feat, dim=0)
#         ca_feat = torch.cat(self.ca_feat, dim=0)
#         ir_feat = torch.cat(self.ir_feat, dim=0)
#         #feat = torch.nn.functional.normalize(feat, dim=1, p=2)
#         ids = torch.cat(self.ids, dim=0)
#         return rgb_feat, ca_feat, ir_feat, ids
# def create_data_generator(model, metrics, device=None):
#     if device:
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#         model.to(device)
#     def _inference(engine, batch):
#         model.eval()
#
#         input10, input11, input2, label1, label2 = batch
#         labels = torch.cat((label1, label1, label2), 0)
#         input10 = Variable(input10.to(device))
#         input11 = Variable(input11.to(device))
#         target = Variable(labels.to(device))
#         input1 = torch.cat((input10, input11,), 0)
#         input2 = Variable(input2.to(device))
#
#         with torch.no_grad():
#             _, feat = model(input1, input2)
#
#             n = feat.shape[0] // 3
#             rgb_feat = feat.narrow(0, 0, n)
#             ca_feat = feat.narrow(0, n, n)
#             ir_feat = feat.narrow(0, 2*n, n)
#             ids = target.narrow(0, 0, n)
#         return rgb_feat.cpu(), ca_feat.cpu(), ir_feat.cpu(), ids.cpu()
#
#     engine = Engine(_inference)
#
#     for name, metric in metrics.items():
#         metric.attach(engine, name)
#     return engine
# def train_fine(epoch):
#     pick_sampler = IdentityPickSampler(trainset.train_color_label, trainset.train_thermal_label,
#                                        color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)
#     trainset.cIndex = pick_sampler.index1  # color index
#     trainset.tIndex = pick_sampler.index2  # thermal index
#     pick_trainloader = data.DataLoader(trainset, batch_size=n_class, sampler=pick_sampler, num_workers=args.workers, drop_last=True)
#
#     data_generator = create_data_generator(net, metrics={'data': data_collector()}, device=device)
#     data_generator.run(pick_trainloader)
#
#     rgb_feats, ca_feats, ir_feats, id = data_generator.state.metrics['data']
#     n_rgb_feat = rgb_feats.chunk(n_class, 0)
#     n_ca_feat = ca_feats.chunk(n_class, 0)
#     n_ir_feat = ir_feats.chunk(n_class, 0)
#
#     for i in range(n_class):
#         center_i = (torch.mean(n_rgb_feat[i], dim=0, keepdim=True) + torch.mean(n_ca_feat[i], dim=0, keepdim=True) + torch.mean(n_ir_feat[i], dim=0, keepdim=True)) / 3.0
#         memory[id[i]:id[i]+1] = center_i.cuda()
# class MemoryTripletLoss(nn.Module):
#     def __init__(self):
#         super(MemoryTripletLoss, self).__init__()
#         self.ranking_loss = nn.MarginRankingLoss(margin=0.3)
#
#     def forward(self, feats, labels):
#         """
#         Args:
#         - inputs: feature matrix with shape (batch_size, feat_dim)
#         - targets: ground truth labels with shape (num_classes)
#         """
#         label_uni = labels.unique()
#         label_num = len(label_uni)
#         feat = feats.chunk(label_num * 3, 0)
#         center = []
#         for i in range(label_num * 3):
#             center.append(torch.mean(feat[i], dim=0, keepdim=True))
#         inputs = torch.cat(center)
#
#         inputs = inputs.chunk(3, 0)
#         rgb_ = inputs[0]
#         ca_ = inputs[1]
#         ir_ = inputs[2]
#         centers = []
#         for i in range(label_num):
#             centers.append((rgb_[i:i+1] + ca_[i:i+1] + ir_[i:i+1]) / 3.0)
#         centers = torch.cat(centers)
#
#         dist = pdist_torch(centers, memory)
#
#         pos_mask = torch.zeros_like(dist)
#         for i in range(label_num):
#             pos_mask[i][label_uni[i]] = 1
#         neg_mask = 1 - pos_mask
#
#         dist_ap, dist_an = [], []
#         for i in range(label_num):
#             dist_ap.append(dist[i][label_uni[i]].unsqueeze(0))
#             mask = neg_mask[i]
#             neg_ = dist[i][torch.where(mask == 1)[0]]
#             dist_an.append(neg_.min().unsqueeze(0))
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
#
#         y = dist_ap.new().resize_as_(dist_ap).fill_(1)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         return loss

def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    metric_loss = AverageMeter()
    metric_loss1 = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(trainloader):
        labels = torch.cat((label1, label1, label2), 0)
        input10 = Variable(input10.cuda())
        input11 = Variable(input11.cuda())
        labels = Variable(labels.cuda())
        # input1 = torch.cat((input10, input11,), 0)
        input2 = Variable(input2.cuda())
        input2 = torch.cat((input2, input11,), 0)
        input1 = Variable(input10.cuda())

        feat, out0, feat_all, feat1, out1 = net(input1, input2)

        loss_id = criterion_id(out0[0], labels) + criterion_id(out1, labels)
        loss_tri_l, batch_acc = criterion_tri(feat[0], labels)
        loss_metric = criterion_metric(feat[0], labels) + criterion_metric1(out0[0], labels)
        for i in range(len(feat) - 1):
            loss_id += criterion_id(out0[i + 1], labels)
            loss_tri_l += criterion_tri(feat[i + 1], labels)[0]
            loss_metric += criterion_metric(feat[i + 1], labels) + criterion_metric1(out0[i + 1], labels)
        loss_tri, batch_acc = criterion_tri(feat_all, labels)
        loss_tri += criterion_tri(feat1, labels)[0]
        loss_tri += loss_tri_l * args.w_center
        correct += batch_acc
        loss = loss_id + loss_tri + loss_metric

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri, 2 * input1.size(0))
        metric_loss.update(loss_metric, 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'M1Loss: {metric_loss.val:.4f} ({metric_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, metric_loss=metric_loss))

    # writer.add_scalar('total_loss', train_loss.avg, epoch)
    # writer.add_scalar('id_loss', id_loss.avg, epoch)
    # writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    # writer.add_scalar('lr', current_lr, epoch)

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    if args.pcb == 'on':
        feat_dim = args.num_strips * args.local_feat_dim
        # feat_dim = 2048
    else:
        feat_dim = 2048
    gall_feat = np.zeros((ngall, feat_dim))
    gall_feat_att = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat = net(input, input, test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, feat_att = net(input, input, test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat = np.zeros((nquery, feat_dim))
    query_feat_att = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, feat_att = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    if args.re_rank == 'random_walk':
        distmat = random_walk(query_feat, gall_feat)
        if args.pcb == 'off': distmat_att = random_walk(query_feat_att, gall_feat_att)
    elif args.re_rank == 'k_reciprocal':
        distmat = k_reciprocal(query_feat, gall_feat)
        if args.pcb == 'off': distmat_att = k_reciprocal(query_feat_att, gall_feat_att)
    elif args.re_rank == 'no':
        # compute the similarity
        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        if args.pcb == 'off': distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
        if args.pcb == 'off':
            cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        if args.pcb == 'off':
            cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    # writer.add_scalar('rank1', cmc[0], epoch)
    # writer.add_scalar('mAP', mAP, epoch)
    # writer.add_scalar('mINP', mINP, epoch)
    if args.pcb == 'off':
        # writer.add_scalar('rank1_att', cmc_att[0], epoch)
        # writer.add_scalar('mAP_att', mAP_att, epoch)
        # writer.add_scalar('mINP_att', mINP_att, epoch)

        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att
    else:
        return cmc, mAP, mINP

best_rank1 = 0.0
best_mAP = 0.0
best_mINP = 0.0
rank1 = 0.0
mAP = 0.0
mINP = 0.0
# training
print('==> Start Training...')
for epoch in range(start_epoch, 61 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label,
                              color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, num_workers=4, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))
        # testing
        cmc_fc, mAP_fc, mINP_fc = test(epoch)
        # save model
        if (cmc_fc[0] > best_rank1 or mAP_fc > best_mAP or mINP_fc > best_mINP) and (cmc_fc[0] + mAP_fc + mINP_fc >= best_rank1 + best_mAP + best_mINP):  # not the real best for sysu-mm01
            if cmc_fc[0] > best_rank1:
                best_rank1 = cmc_fc[0]
            if mAP_fc > best_mAP:
                best_mAP = mAP_fc
            if mINP_fc > best_mINP:
                best_mINP = mINP_fc

            best_epoch = epoch
            rank1 = cmc_fc[0]
            mAP = mAP_fc
            mINP = mINP_fc
            state = {
                'net': net.state_dict(),
                'cmc': cmc_fc,
                'mAP': mAP_fc,
                'mINP': mINP_fc,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
        print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, rank1, mAP, mINP))

