import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import torch
from torchvision import transforms
from torch.autograd import Variable

from model.vgg_siamese2 import vgg16_basenet, SiameseNetBaseline
from libs.firearm_data import QuerySet
from libs import custom_transform
from util.eval_metric import average_precision, recall_at_k


def main():
    position = [1, 2, 4, 8, 16, 32]
    feature_dim = [16, 32, 64, 128, 256, 512]

    torch.cuda.set_device(0)

    # do not do data augmentation on validation set
    val_trans = transforms.Compose([
        custom_transform.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dir = "data/firearm-dataset/test"
    val_set = QuerySet(root=val_dir, transform=val_trans)

    cls_model_dir = "model/checkpoint/vgg16_cls/model_best.pth.tar"
    embed_net = vgg16_basenet(pretrained=True, checkpoint_dir=cls_model_dir)
    sim_net = SiameseNetBaseline(embed_net).cuda()

    model_dir = "model/checkpoint/vgg16_retr_from_cls/model_best.pth.tar"
    checkpoint = torch.load(model_dir)
    sim_net.load_state_dict(checkpoint['state_dict'])

    sim_net.eval()
    database_feat = []

    if os.path.exists("database_feat.pth"):
        database_feat = torch.load("database_feat.pth")
    else:
        for i in range(len(val_set.database_imgs)):
            img = val_set.get_img(i, query=False)
            img = Variable(img.unsqueeze(0).cuda(), volatile=True)
            feat = sim_net.forward_once(img)
            database_feat.append(feat.data)

        database_feat = torch.cat(database_feat, dim=0)
        torch.save(database_feat, "database_feat.pth")

    # database_feat size: N*D (N is sample number, D is feature dim)
    database_feat = database_feat.cpu().numpy()

    for dim in feature_dim:
        # pca on database feature
        pca = PCA(n_components=dim, whiten=False)
        new_feat = pca.fit_transform(database_feat)
        new_feat = normalize(new_feat)

        aps = []
        scores = np.zeros(len(position))

        for query_id in range(len(val_set.query_imgs)):

            query_img = val_set.get_img(query_id, query=True)
            query_img = Variable(query_img.unsqueeze(0).cuda(), volatile=True)
            query_feat = sim_net.forward_once(query_img)

            # pca on query feature
            query_feat = pca.transform(query_feat.data.cpu().numpy())
            query_feat = normalize(query_feat)

            similarity = (query_feat * new_feat).sum(axis=1)
            # get idx in descending order of similarity
            idx = similarity.argsort()[::-1]

            ap = average_precision(list(idx), val_set.gt_info[query_id])
            aps.append(ap)

            # print("query {}, ap is {:.3f}".format(query_id, ap))

            single_score = recall_at_k(position, list(idx), val_set.gt_info[query_id])
            scores += np.array(single_score)

        scores /= len(val_set.query_imgs)

        print("feature dim: {}, mAP: {}".format(dim, sum(aps)/len(aps)))
        print("Recall@k: {}".format(scores))

        # aps = np.array(aps)
        # np.save("comparison_feature/siamese_firearm_cls_finetuned_vgg", aps)


if __name__ == "__main__":
    main()