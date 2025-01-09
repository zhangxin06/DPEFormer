import numpy as np
import torch
import cv2
import random
import os
import torchvision.transforms as T
import json

def paste_on_person(mask_image, person_image):
    mask_h, mask_w = mask_image.shape[:2]
    image_h, image_w = person_image.shape[:2]
    mask_scale = mask_h/mask_w

    if mask_scale > 2:
        re_size = (random.randint(image_w//2, image_w*3//4), image_h)
    else:
        re_size = (image_w, random.randint(image_h//2, image_h*3//4))
    paste_image = cv2.resize(mask_image, re_size)
    paste_image = cv2.flip(paste_image, 1)


    w_, h_ = re_size[0], re_size[1]
    index = random.randint(0, 3)
    if index == 0:
        h_begin, w_begin = 0, 0
    elif index == 1:
        h_begin, w_begin = 0, image_w-w_
    elif index == 2:
        h_begin, w_begin = image_h-h_, 0
    else:
        h_begin, w_begin = image_h-h_, image_w - w_

    mask = np.zeros_like(person_image)
    mask[h_begin:h_begin + h_, w_begin:w_begin + w_, :] = paste_image


    mask_person = (mask==0)

    result = person_image*mask_person+mask
    return result




class SAM_model(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root

        self.occ_imgs = os.listdir('/home/brl/BRL/ZX/Datasets/SA-1B')

        after = filter(lambda s: s.endswith(".json"), self.occ_imgs)
        self.occ_imgs = list(after)

        self.occ_imgs = self.occ_imgs[:10]

        self.len = len(self.occ_imgs)



    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        img = img.permute(1,2,0)
        img = img.numpy()

        index = random.randint(0, 10 - 1)

        json_name = self.occ_imgs[index]

        path = os.path.join('/home/brl/BRL/ZX/Datasets/SA-1B', json_name)
        with open(path) as f:
            results = json.load(f)

        filename = results['image']['file_name']
        all_annotations = results['annotations']
        index_in_each_json = random.randint(0, len(all_annotations) - 1)

        image_path = os.path.join('/home/brl/BRL/ZX/Datasets/SA-1B', filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation = all_annotations[index_in_each_json]['segmentation']
        size = segmentation['size']
        counts = segmentation['counts']
        rle = {'size': size, 'counts': counts}

        import pycocotools.mask as mask_util

        mask = mask_util.decode(rle)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        non_zero_pixels = np.transpose(np.nonzero(mask))
        x1, y1 = np.min(non_zero_pixels, axis=0)
        x2, y2 = np.max(non_zero_pixels, axis=0)
        cropped_image = masked_image[x1:x2 + 1, y1:y2 + 1]
        function = T.Compose([
            T.ToTensor(),
        ])
        cropped_image = function(cropped_image)
        cropped_image = cropped_image.permute(1,2,0)
        cropped_image = cropped_image.numpy()

        sam_paste_image = paste_on_person(mask_image=cropped_image, person_image=img)
        sam_paste_image = torch.from_numpy(sam_paste_image)
        sam_paste_image = sam_paste_image.permute(2,0,1)

        return sam_paste_image


class SAM_model_no_json(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        self.occ_imgs = os.listdir('/home/fabian/BRL/zhangxin/Datasets/SA-1B-crop')
        self.length = len(self.occ_imgs)



    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img

        img = img.permute(1,2,0)
        img = img.numpy()

        index = random.randint(0, self.length - 1)
        crop_dir = self.occ_imgs[index]

        crop_img_list = os.listdir(os.path.join('/home/fabian/BRL/zhangxin/Datasets/SA-1B-crop', crop_dir))
        length_in_each_crop = len(crop_img_list)
        index = random.randint(0, length_in_each_crop - 1)
        cropped_image_name = crop_img_list[index]

        file_name = os.path.join('/home/fabian/BRL/zhangxin/Datasets/SA-1B-crop', crop_dir, cropped_image_name)

        cropped_image = cv2.imread(file_name)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        function = T.Compose([
            T.ToTensor(),
        ])
        cropped_image = function(cropped_image)
        cropped_image = cropped_image.permute(1,2,0)
        cropped_image = cropped_image.numpy()

        sam_paste_image = paste_on_person(mask_image=cropped_image, person_image=img)
        sam_paste_image = torch.from_numpy(sam_paste_image)
        sam_paste_image = sam_paste_image.permute(2,0,1)

        return sam_paste_image