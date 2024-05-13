import copy
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
from torch.nn.parameter import Parameter
import random
import math
from IPython import embed

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

# 通道注意力机制
class ChannelAttention1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# class ChannelAttention2(nn.Module):
#     def __init__(self, in_planes, ratio=4):
#         super(ChannelAttention2, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#
#
#     def forward(self, x):
#         out = self.relu1(self.fc1(self.avg_pool(x)))
#         return out

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# # 空间注意力机制
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
# # AFAM注意力模块
# class AFAM(nn.Module):
#     def __init__(self, c1, c2):
#         super(AFAM, self).__init__()
#         self.channel_attention = ChannelAttention(c1)
#         self.spatial_attention = SpatialAttention()
#
#         self.non_local = Non_local(c1)
#
#     def forward(self, x):
#         c_mask = self.channel_attention(x) * x
#         s_mask = self.spatial_attention(x) * x
#         out = self.non_local((c_mask + s_mask) / 2.0)
#         # out = self.channel_attention(x) * x
#         # out = self.spatial_attention(out) * out
#         return out
#
#
# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
# def swish(x):
#     return x * torch.sigmoid(x)
# ACT_FNS = {
#     'relu': nn.ReLU,
#     'swish': swish,
#     'gelu': gelu
# }
# class MLP(nn.Module):
#     def __init__(self, n_state):
#         super(MLP, self).__init__()
#         nx = 1536
#         self.c_fc = Conv1D(n_state, 1, nx)
#         self.c_proj = Conv1D(nx, 1, n_state)
#         self.act = ACT_FNS['gelu']
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         h = self.act(self.c_fc(x))
#         h2 = self.c_proj(h)
#         return self.dropout(h2)
# class LayerNorm(nn.Module):
#     "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."
#
#     def __init__(self, n_state, e=1e-5):
#         super(LayerNorm, self).__init__()
#         self.g = nn.Parameter(torch.ones(n_state))
#         self.b = nn.Parameter(torch.zeros(n_state))
#         self.e = e
#
#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.e)
#         return self.g * x + self.b
# class Conv1D(nn.Module):
#     def __init__(self, nf, rf, nx):
#         super(Conv1D, self).__init__()
#         self.rf = rf
#         self.nf = nf
#         if rf == 1:  # faster 1x1 conv
#             w = torch.empty(nx, nf)   # [256, 768/256]
#             nn.init.normal_(w, std=0.02)
#             self.w = Parameter(w)
#             self.b = Parameter(torch.zeros(nf))
#         else:  # was used to train LM
#             raise NotImplementedError
#
#     def forward(self, x):
#         if self.rf == 1:
#             size_out = x.size()[:-1] + (self.nf,)
#             x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
#             x = x.view(*size_out)
#         else:
#             raise NotImplementedError
#         return x
# class Attention(nn.Module):
#     def __init__(self, nx, n_ctx, scale=False):
#         super(Attention, self).__init__()
#         n_state = nx  # 256
#         self.n_head = 2 # 2
#         self.split_size = n_state  # 256
#         self.scale = scale  # True
#         self.c_attn = Conv1D(n_state * 3, 1, nx)
#         self.c_proj = Conv1D(n_state, 1, nx)
#
#         self.resid_dropout = nn.Dropout(0.1)
#
#     def _attn(self, q, k, v, num_landmark, rns_indices):
#         """
#         Args:
#             q: [2, 2, 7000, 128]
#             k: [2, 2, 128, 7000]
#             v: [2, 2, 7000, 128]
#             num_landmark: 5
#             rns_indices: [2, 7000, 20]
#
#         Returns:
#             [2, 2, 7000, 128]
#         """
#         data_length = q.shape[2]  # 7000
#         landmark = torch.Tensor(random.sample(range(data_length), num_landmark)).long()  # tensor([a, b, c, d, e])
#
#         sq = q[:, :, landmark, :].contiguous()
#         sk = k[:, :, :, landmark].contiguous()
#
#         w1 = torch.matmul(q, sk)  # [2, 2, 7000, 5]
#         w2 = torch.matmul(sq, k)  # [2, 2, 5, 7000]
#         w = torch.matmul(w1, w2)  # [2, 2, 5, 7000]
#
#         if self.scale:
#             w = w / math.sqrt(v.size(-1))
#         return self.rns(w, v, rns_indices)
#
#     def rns(self, w, v, rns_indices):
#         """
#         Args:
#             w: [2, 2, 7000, 7000]
#             v: [2, 2, 7000, 128]
#             rns_indices: [2, 7000, 20]
#         Returns:
#             a_v: [2, 2, 7000, 128]
#         """
#         bs, hn, dl, _ = w.shape
#         rns_indices = rns_indices.unsqueeze(1).repeat(1, hn, 1, 1)
#         mask = torch.zeros_like(w).scatter_(3, rns_indices, torch.ones_like(rns_indices, dtype=w.dtype))
#         mask = mask * mask.transpose(2, 3)
#         if 'cuda' in str(w.device):
#             mask = mask.cuda()
#         else:
#             mask = mask.cpu()
#         if self.training:
#             w = w * mask + -1e9 * (1 - mask)  # [2, 2, 7000, 7000]
#             w = F.softmax(w, dim=3)  # [2, 2, 7000, 7000]
#             a_v = torch.matmul(w, v)  # [2, 2, 7000, 128]
#         else:
#             w = (w * mask).reshape(bs * hn, dl, dl).to_sparse()
#             w = torch.sparse.softmax(w, 2)
#             v = v.reshape(bs * hn, dl, -1)
#             a_v = torch.bmm(w, v).reshape(bs, hn, dl, -1)
#         return a_v
#
#     def merge_heads(self, x):
#         x = x.permute(0, 2, 1, 3).contiguous()
#         new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
#         return x.view(*new_x_shape)
#
#     def split_heads(self, x, k=False):
#         new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
#         x = x.view(*new_x_shape)
#         if k:
#             return x.permute(0, 2, 3, 1)
#         else:
#             return x.permute(0, 2, 1, 3)
#
#     def forward(self, x, num_landmark, rns_indices):
#         x = self.c_attn(x)
#         query, key, value = x.split(self.split_size, dim=2)
#         query = self.split_heads(query)
#         key = self.split_heads(key, k=True)
#         value = self.split_heads(value)
#         mask = None
#         a = self._attn(query, key, value, num_landmark, rns_indices)  # [2, 2, 7000, 128]
#         a = self.merge_heads(a)  # [2, 7000, 256]
#         a = self.c_proj(a)  # [2, 7000, 256]
#         a = self.resid_dropout(a)  # [2, 7000, 256]
#         return a
# class Block(nn.Module):
#     def __init__(self, n_ctx, scale=False):
#         super(Block, self).__init__()
#         nx = 1536  # 256
#         self.attn = Attention(nx, n_ctx, scale)    # n_ctx=1024
#         self.ln_1 = LayerNorm(nx)
#         self.mlp = MLP(4 * nx)
#         self.ln_2 = LayerNorm(nx)
#
#     def forward(self, x, num_landmark, rns_indices):
#         """
#         Args:
#             x: [2, 7000, 256]
#             num_landmark: 5
#             rns_indices: [2, 7000, 20]
#
#         Returns:
#
#         """
#         a = self.attn(x, num_landmark, rns_indices)   # [2, 7000, 256]
#         n = self.ln_1(x + a)  # [2, 7000, 256]
#         m = self.mlp(n)  # [2, 7000, 256]
#         h = self.ln_2(n + m)  # [2, 7000, 256]
#         return h

# #####################################################################

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer' + str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.thermal, 'layer' + str(i), getattr(model_t, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.thermal, 'layer' + str(i))(x)
            return x

class mix_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(mix_module, self).__init__()

        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer' + str(i))(x)
            return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base, 'layer' + str(i), getattr(model_base, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer' + str(i))(x)
            return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='off', gm_pool='on', arch='resnet50', share_net=1, pcb='on',
                 local_feat_dim=256, num_strips=6, p=10):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        self.base_resnet = base_resnet(arch=arch, share_net=share_net)

        self.p = p
        self.non_local = no_local
        self.pcb = pcb
        if self.non_local == 'on':
            pass

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool

        if self.pcb == 'on':
            self.num_stripes = num_strips
            local_conv_out_channels = local_feat_dim

            self.local_conv_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))

            self.fc_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)

            block = ChannelAttention1(2048)
            self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(self.num_stripes)])

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x1, x2=None, modal=1):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        # shared block
        if self.non_local == 'on':
            pass
        else:
            x = self.base_resnet(x)

        b, c, h, w = x.shape
        xx = x.view(b, c, -1)
        x_pool = (torch.mean(xx ** self.p, dim=-1) + 1e-12) ** (1 / self.p)
        g_feat = self.bottleneck(x_pool)

        return g_feat



