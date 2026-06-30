import torch.utils.data as data
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps # 确保 ImageOps 引入，用于填充
import random
import torch

# ======================================================
#  sRGB → Linear RGB 转换函数（新增，仅影响输入端）
# ======================================================
def srgb_to_linear(t):
    """ t: a tensor image in [0,1], shape (C,H,W) """
    return torch.where(
        t <= 0.04045,
        t / 12.92,
        ((t + 0.055) / 1.055) ** 2.4
    )


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, label_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames

        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.label_filenames[index])

        input, target, _ = get_patch(input, target, self.patch_size)

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        # ============================
        # ⭐ transform 后做 sRGB → linear
        # ============================
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

            input = srgb_to_linear(input)
            target = srgb_to_linear(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames

        self.transform = transform

    def __getitem__(self, index):
        # 加载图像
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.label_filenames[index])

        # --- 针对评估模式的零填充修改开始 ---
        factor = 2 # 适配 PixelUnshuffle (factor=2)
        w, h = input.size
        
        # 计算需要填充的像素量，确保 H 和 W 都是 factor 的倍数
        H_pad = (factor - h % factor) % factor
        W_pad = (factor - w % factor) % factor
        
        # 记录原始尺寸，供评估脚本裁剪回原图使用
        original_size = (h, w) # (H, W)
        
        if H_pad > 0 or W_pad > 0:
            # 零填充操作: ImageOps.expand(image, border=(左, 上, 右, 下), fill=0)
            # 我们只需要在底部 (H) 和右侧 (W) 填充
            border = (0, 0, W_pad, H_pad)
            
            # 对输入和目标图像应用相同的零填充
            input = ImageOps.expand(input, border=border, fill=0)
            target = ImageOps.expand(target, border=border, fill=0)
        
        # --- 零填充修改结束 ---

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

            # ⭐ 加这里
            input = srgb_to_linear(input)
            target = srgb_to_linear(target)

        # 返回时带上原始尺寸信息，供评估脚本在计算指标时裁剪模型输出使用
        return input, target, file, original_size

    def __len__(self):
        return len(self.label_filenames)