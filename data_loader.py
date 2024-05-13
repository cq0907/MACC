import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import math
import matplotlib.pylab as plt
from IPython import embed

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img

class ChannelExchange1(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[:, :, 1] = img[:, :, 0]
            img[:, :, 2] = img[:, :, 0]
        elif idx == 1:
            # random select B Channel
            img[:, :, 0] = img[:, :, 1]
            img[:, :, 2] = img[:, :, 1]
        elif idx == 2:
            # random select G Channel
            img[:, :, 0] = img[:, :, 2]
            img[:, :, 1] = img[:, :, 2]
        return img

class ChannelAdap(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            img = img

        return img

class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img

class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):

        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

class MixPatch(object):
    def __init__(self, patch_size=16, ratios=0.9):
        self.patch_size = patch_size
        self.ratios = ratios

    def __call__(self, v_img, t_img):
        h, w, c = v_img.shape
        num_patchs = (h / self.patch_size) * (w / self.patch_size)
        v_patchs = int(num_patchs * self.ratios)
        t_patchs = int(num_patchs - v_patchs)

        count = 0
        v_flags = torch.ones((v_patchs,))
        t_flags = torch.zeros((t_patchs,))
        flags = torch.concat([v_flags, t_flags])
        idx = random.sample(range(len(flags)), len(flags))
        img = np.zeros_like(v_img)
        for i in range(int((h / self.patch_size))):
            for j in range(int(w / self.patch_size)):
                if flags[idx[count]] == 0:
                    img[i*self.patch_size: (i+1)*self.patch_size, j*self.patch_size: (j+1)*self.patch_size, :] = v_img[i*self.patch_size: (i+1)*self.patch_size, j*self.patch_size: (j+1)*self.patch_size, :]
                elif flags[idx[count]] == 1:
                    img[i*self.patch_size: (i+1)*self.patch_size, j*self.patch_size: (j+1)*self.patch_size, :] = t_img[i*self.patch_size: (i+1)*self.patch_size, j*self.patch_size: (j+1)*self.patch_size, :]
                count += 1
        return img

class MixPatch1(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.gray = ChannelExchange(gray=2)


    def __call__(self, v_img, t_img):
        mix_img = self.gray(np.copy(v_img))
        for attempt in range(100):
            area = v_img.shape[0] * v_img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < v_img.shape[1] and h < v_img.shape[0]:
                x1 = random.randint(0, v_img.shape[0] - h)
                y1 = random.randint(0, v_img.shape[1] - w)
                if v_img.shape[2] == 3:
                    mix_img[x1:x1 + h, y1:y1 + w, 0] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 0])
                    mix_img[x1:x1 + h, y1:y1 + w, 1] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 1])
                    mix_img[x1:x1 + h, y1:y1 + w, 2] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 2])
                else:
                    mix_img[x1:x1 + h, y1:y1 + w, 0] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 1])
                break
        return mix_img

class MixPatch2(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, v_img, t_img):
        # for attempt in range(100):
        #     area = v_img.shape[0] * v_img.shape[1]
        #
        #     target_area = random.uniform(self.sl, self.sh) * area
        #     aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        #
        #     h = int(round(math.sqrt(target_area * aspect_ratio)))
        #     w = int(round(math.sqrt(target_area / aspect_ratio)))
        #
        #     if w < v_img.shape[1] and h < v_img.shape[0]:
        #         x1 = random.randint(0, v_img.shape[0] - h)
        #         y1 = random.randint(0, v_img.shape[1] - w)
        #
        #         x2 = random.randint(0, v_img.shape[0] - h)
        #         y2 = random.randint(0, v_img.shape[1] - w)
        #
        #         v_temp = np.copy(v_img[x1:x1 + h, y1:y1 + w, :])
        #         t_temp = np.copy(t_img[x2:x2 + h, y2:y2 + w, :])
        #
        #         v_img[x2:x2 + h, y2:y2 + w, :] = t_temp
        #         t_img[x1:x1 + h, y1:y1 + w, :] = v_temp
        #
        #
        #         return v_img, t_img
        # return v_img, t_img
        mix_img = np.copy(v_img)
        if random.uniform(0, 1) > self.probability:
            # return mix_img
            for attempt in range(100):
                area = v_img.shape[0] * v_img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < v_img.shape[1] and h < v_img.shape[0]:
                    x1 = random.randint(0, v_img.shape[0] - h)
                    y1 = random.randint(0, v_img.shape[1] - w)

                    mix_img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    mix_img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    mix_img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]

                    return mix_img
            return mix_img

        for attempt in range(100):
            area = v_img.shape[0] * v_img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < v_img.shape[1] and h < v_img.shape[0]:
                x1 = random.randint(0, v_img.shape[0] - h)
                y1 = random.randint(0, v_img.shape[1] - w)
                if v_img.shape[2] == 3:
                    mix_img[x1:x1 + h, y1:y1 + w, 0] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 0])
                    mix_img[x1:x1 + h, y1:y1 + w, 1] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 1])
                    mix_img[x1:x1 + h, y1:y1 + w, 2] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 2])
                else:
                    mix_img[x1:x1 + h, y1:y1 + w, 0] = np.copy(t_img[x1:x1 + h, y1:y1 + w, 1])
                return mix_img
        return mix_img


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, sh=0.4):
        data_dir = '../data/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.mix_patch = MixPatch2(sh=sh)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ChannelRandomErasing(probability=0.5),
            normalize,
            ChannelAdapGray(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            ChannelRandomErasing(probability=0.5),
            normalize])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            # ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        mix_img = self.mix_patch(img1, img2)

        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(mix_img)
        img2 = self.transform_thermal(img2)
        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None, sh=0.4):
        # Load training images (path) and labels
        data_dir = '../data/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.mix_patch = MixPatch1(sh=sh)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            ChannelRandomErasing(probability=0.5),
            normalize])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            # ChannelRandomErasing(probability=0.5),
            # ChannelExchange(gray=2),
        ])

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        mix_img = self.mix_patch(img1, img2)

        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(mix_img)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)

        return img1, target1

    def __len__(self):
        return len(self.test_image)


class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class PCBDataset(data.Dataset):
    def __init__(self, data, data_length = 1024):
        self.rgb_feat, self.ca_feat, self.ir_feat, self.ids = data
        self.data_length = data_length
        self.data_num = self.rgb_feat.shape[0]

    def __len__(self):
        return self.rgb_feat.shape[0]

    def __getitem__(self, index):
        pass

        # return feat_rgb_, feat_ca_, feat_ir_, id_rgb_, id_ca_, id_ir_


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label