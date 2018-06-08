import os
from os.path import isdir, join
import json
from PIL import Image
from glob import glob
import numpy as np
import random
from operator import itemgetter

import torch.utils.data


def my_collate(batch):
    data1, data2, target = list(zip(*batch))
    target = torch.FloatTensor(target)
    return [data1, data2, target]


def triplet_collate(batch):
    data1, data2, data3 = list(zip(*batch))

    return [data1, data2, data3]


def my_collate_cls(batch):
    data, target = list(zip(*batch))
    target = torch.LongTensor(target)
    return [data, target]


def find_class(root):
    classes = [d for d in os.listdir(root) if isdir(join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, class_to_idx):
    images = []
    labels = []

    root = os.path.expanduser(root)

    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)

        if not os.path.isdir(d):
            continue

        im_path = glob(os.path.join(d, "*.jpg"))
        for path in sorted(im_path):
            images.append(path)
            labels.append(class_to_idx[target])

    return images, labels


class FirearmDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_pair_per_class,
                 transform=None):
        classes, class_to_idx = find_class(root)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.img_pair_per_class = img_pair_per_class
        self.transform = transform

        imgs, labels = make_dataset(root, class_to_idx)
        self.imgs = imgs
        self.labels = labels

        img_pair = self._make_img_pair()
        self.img_pair = img_pair

    def __getitem__(self, index):
        im_pth1, im_pth2, label = self.img_pair[index]
        im1 = Image.open(im_pth1).convert("RGB")
        im2 = Image.open(im_pth2).convert("RGB")

        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        return im1, im2, label

    def _make_img_pair(self):
        img_pair = []
        labels = np.asarray(self.labels)

        all_cls_img_idx = {}
        for i in range(len(self.classes)):
            all_cls_img_idx[i] = list(np.where(labels == i)[0])

        # for each category we choose im_pair_per_class//2 positive pair
        # and equal number of negative pair
        for i in range(len(self.classes)):
            # choose positive pair
            positive_img_idx = all_cls_img_idx[i]

            for j in range(self.img_pair_per_class//2):
                idx1, idx2 = random.sample(positive_img_idx, k=2)

                pth1, pth2 = itemgetter(idx1, idx2)(self.imgs)
                img_pair.append((pth1, pth2, 1.0))

            other_cls_idx = list(range(len(self.classes)))
            other_cls_idx.remove(i)

            # choose negative pair
            for j in range(self.img_pair_per_class//2):
                idx1 = random.choice(positive_img_idx)

                neg_cls_idx = random.choice(other_cls_idx)
                negative_img_idx = all_cls_img_idx[neg_cls_idx]
                idx2 = random.choice(negative_img_idx)

                pth1, pth2 = itemgetter(idx1, idx2)(self.imgs)
                img_pair.append((pth1, pth2, 0.0))

        random.shuffle(img_pair)
        return img_pair

    # update the image pair list every N training epoch
    # to create more dataset variability
    def regenerate_img_pair(self):
        self.img_pair = self._make_img_pair()

    def __len__(self):
        return len(self.img_pair)


class FirearmTriplet(torch.utils.data.Dataset):
    def __init__(self, root, triplet_per_class, transform=None):
        self.root = root
        classes, class_to_idx = find_class(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.triplet_per_class = triplet_per_class
        self.transform = transform

        imgs, labels = make_dataset(root, class_to_idx)
        self.imgs = imgs
        self.labels = labels

        self.triplet_list = self._make_triplet()

    def _make_triplet(self):
        triplet_list = []
        labels = np.asarray(self.labels)

        # pre-compute img index for each class
        all_cls_img_idx = {}
        for i in range(len(self.classes)):
            all_cls_img_idx[i] = list(np.where(labels==i)[0])

        for i in range(len(self.classes)):
            # generate triplet_per_class triplet for each class
            cls_img_index = all_cls_img_idx[i]

            other_cls_idx = list(range(len(self.classes)))
            other_cls_idx.remove(i)

            for j in range(self.triplet_per_class):
                anc_idx, pos_idx = random.sample(cls_img_index, k=2)

                # sample a negative image from other class randomly

                neg_cls_idx = random.choice(other_cls_idx)
                negative_img_idx = all_cls_img_idx[neg_cls_idx]

                neg_idx = random.choice(negative_img_idx)

                pth1, pth2, pth3 = itemgetter(anc_idx, pos_idx, neg_idx)(self.imgs)
                triplet_list.append((pth1, pth2, pth3))

        random.shuffle(triplet_list)

        return triplet_list

    def __getitem__(self, index):
        pth1, pth2, pth3 = self.triplet_list[index]

        anchor = Image.open(pth1).convert('RGB')
        positive = Image.open(pth2).convert('RGB')
        negative = Image.open(pth3).convert('RGB')

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def regenerate_triplet(self):
        self.triplet_list = self._make_triplet()

    def __len__(self):
        return len(self.triplet_list)


class QuerySet(object):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.query_imgs = self._make_query_img()
        self.database_imgs = self._make_database_img()
        self.gt_info = self._make_gt_info()

    def _make_query_img(self):
        query_img_dir = join(self.root, "query_info", "query_image")
        query_imgs = glob(join(query_img_dir, "*.jpg"))
        query_imgs.sort()
        return query_imgs

    def _make_database_img(self):
        database_img_dir = join(self.root, "database_image")
        database_imgs = glob(join(database_img_dir, "*.jpg"))
        database_imgs.sort()
        return database_imgs

    def _make_gt_info(self):
        gt_file = join(self.root, "query_info", "ground_truth_info.json")

        # a list of list, new_gt_info[i] is the ground truth image indices
        # for ith query
        new_gt_info = []
        with open(gt_file, "r") as f:
            ori_gt_info = json.load(f)

        # imgs without path info
        query_imgs = [os.path.basename(x) for x in self.query_imgs]
        database_imgs = [os.path.basename(x) for x in self.database_imgs]

        for i, im in enumerate(query_imgs):
            gt_im = ori_gt_info[im]
            gt_im_id = [database_imgs.index(x) for x in gt_im]
            new_gt_info.append(gt_im_id)

        return new_gt_info

    def get_img(self, index, query=False):
        if query:
            im_path = self.query_imgs[index]
        else:
            im_path = self.database_imgs[index]

        img = Image.open(im_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img


class FirearmCls(torch.utils.data.Dataset):
    def __init__(self, root, ratio=0.7, train=True,
                 regen_train_test=True, transform=None):
        self.root = root
        self.ratio = ratio
        self.train= train
        self.regen_train_test = regen_train_test
        self.transform = transform

        self.classes, self.class_to_idx = find_class(self.root)
        self.imgs, self.labels = make_dataset(self.root, self.class_to_idx)

        if self.regen_train_test:
            _train, _test = self._split_train_test()
            self.train_imgs, self.train_labels = _train
            self.test_imgs, self.test_labels = _test
        else:
            train_pth = "/home/jdhao/PycharmProjects/firearm_retrieval_experiment/libs/cls_train_data.pt"
            test_pth = "/home/jdhao/PycharmProjects/firearm_retrieval_experiment/libs/cls_test_data.pt"
            self.train_imgs, self.train_labels = torch.load(train_pth)
            self.test_imgs, self.test_labels = torch.load(test_pth)

        self.class_weight = self._cal_class_weight()

    def _split_train_test(self):
        labels = np.asarray(self.labels)

        num_cls = np.unique(self.labels).size
        train_imgs = []
        train_labels = []
        test_imgs = []
        test_labels = []

        for cls in range(num_cls):
            idxs = np.where(labels == cls)[0]
            train_num = int(len(idxs) * self.ratio)
            train_idx = random.sample(list(idxs), train_num)
            test_idx = [x for x in idxs if x not in train_idx]

            train_imgs.extend(itemgetter(*train_idx)(self.imgs))
            train_labels.extend(itemgetter(*train_idx)(self.labels))
            test_imgs.extend(itemgetter(*test_idx)(self.imgs))
            test_labels.extend(itemgetter(*test_idx)(self.labels))

        # we should shuffle the data before saving
        train_data = list(zip(train_imgs, train_labels))
        random.shuffle(train_data)
        train_imgs, train_labels = zip(*train_data)

        test_data = list(zip(test_imgs, test_labels))
        random.shuffle(test_data)
        test_imgs, test_labels = zip(*test_data)

        torch.save((train_imgs, train_labels), "cls_train_data.pt")
        torch.save((test_imgs, test_labels), "cls_test_data.pt")

        return (train_imgs, train_labels), (test_imgs, test_labels)

    def _cal_class_weight(self):
        if self.train:
            cls_num = []
            train_labels = np.asarray(self.train_labels)
            num_cls = np.unique(train_labels).size

            for i in range(num_cls):
                cls_num.append(len(np.where(train_labels==i)[0]))

            cls_num = np.array(cls_num)
            # print(cls_num.shape)

            cls_num = cls_num.max()/cls_num
            x = 1.0/np.sum(cls_num)

            return x*cls_num
        else:
            pass

    def __getitem__(self, index):
        if self.train:
            img = Image.open(self.train_imgs[index]).convert('RGB')
            target = self.train_labels[index]

            if self.transform:
                img = self.transform(img)

        else:
            img = Image.open(self.test_imgs[index]).convert('RGB')
            target = self.test_labels[index]

            if self.transform:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)
