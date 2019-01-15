import argparse
import time
import os
import shutil
import json

import torch
from torchvision import transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.vgg_siamese2 import vgg16_basenet, SiameseNetBaseline
from libs.firearm_data import FirearmDataset, QuerySet, my_collate
from libs import custom_transform
from libs.custom_module import SoftContrastiveLoss
from util.eval_metric import average_precision

parser = argparse.ArgumentParser(description="Firearm Retrieval Baseline")

parser.add_argument("--exp-name", type=str, default="vgg16_retr_from_cls",
                    help="experiment name (default: none)")
parser.add_argument("--batch-size", type=int, default=64,
                    help="batch size for training (default: 64)")
parser.add_argument("--real_batchsize", type=int, default=32,
                    help="real batch size for backward (default: 32)")
parser.add_argument("--epochs", type=int, default=30,
                    help="number of epochs to train (default: 30)")
parser.add_argument("--start-epoch", type=int, default=0,
                    help="manual restart epoch number (default: 0)")
parser.add_argument("--resume", type=str, default="",
                    help="path to the latest checkpoint (default: none)")
parser.add_argument("--learning-rate", type=float, default=0.001,
                    help="initial learning rate (defaut: 0.001)")
parser.add_argument("--momentum", type=float, default=0.9,
                    help="momentum for SGD (default: 0.9)")
parser.add_argument("--weight-decay", type=float, default=0.0005,
                    help="weight decay parameter (default: 0.0005)")
parser.add_argument("--margin1", type=float, default=0.8,
                    help="margin for contrastive loss (default: 0.8)")
parser.add_argument("--margin2", type=float, default=1.2,
                    help="margin for contrastive loss (default: 1.2)")
parser.add_argument("--data", type=str, default="data/firearm-dataset",
                    help="dataset root (default: data/firearm-dataset)")
parser.add_argument("--img-pair-per-class", type=int, default=360,
                    help="number of training image pair generated for"
                         "each class (default: 360)")
parser.add_argument("--gpu-id", type=int, default=2, choices=[0, 1, 2, 3],
                    help="GPU to use (default: 2)")
parser.add_argument("--worker", type=int, default=6,
                    help="number of workers to fetch the data")
parser.add_argument("--print-freq", type=int, default=20,
                    help="training stats print frequency (default: 20)")

args = parser.parse_args()
best_mAP = 0

train_loss = []
val_mAP = []
torch.cuda.set_device(args.gpu_id)  # set the gpu id to use
# use imagenet mean for now
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# normalize = transforms.Normalize(mean=[0.57889945, 0.54142576, 0.51150703],
#                                  std=[0.30337277, 0.30145828, 0.31259883])

# do not do data augmentation on validation set
val_trans = transforms.Compose([
    custom_transform.Resize(size=384),
    transforms.ToTensor(),
    normalize])

val_dir = os.path.join(args.data, "validation")
val_set = QuerySet(root=val_dir, transform=val_trans)


def main():
    global args, best_mAP, val_mAP

    embed_net = vgg16_basenet(pretrained=True,
                              checkpoint_dir="model/checkpoint/vgg16_cls/"
                              "model_best.pth.tar")
    sim_net = SiameseNetBaseline(embed_net).cuda()

    criterion = SoftContrastiveLoss(margin1=args.margin1, margin2=args.margin2)
    criterion.cuda()

    optimizer = optim.SGD(sim_net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            sim_net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    transform = transforms.Compose([
        custom_transform.RandomResize(min_size=256, max_size=384),
        transforms.RandomHorizontalFlip(),
        custom_transform.RandomRotate(5),
        custom_transform.ColorJitter(0, 0.5, 0.5),
        transforms.ToTensor(),
        normalize])

    traindir = os.path.join(args.data, "train")
    train_set = FirearmDataset(root=traindir,
                               img_pair_per_class=args.img_pair_per_class,
                               transform=transform)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=my_collate,
                              num_workers=args.worker, pin_memory=True)

    print("training start!\n")
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0 and epoch % 5 == 0:
            train_set.regenerate_img_pair()

        adjust_learning_rate(optimizer, epoch)

        # train the network for 1 epoch
        train(sim_net, train_loader, criterion, optimizer, epoch)
        cur_map = validate(sim_net)
        val_mAP.append(cur_map)

        is_best = cur_map > best_mAP
        best_mAP = max(cur_map, best_mAP)
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": sim_net.state_dict(),
                         "best_mAP": best_mAP,
                         "optimizer": optimizer.state_dict(),
                         }, is_best)

    log_dir = os.path.join("result", args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, "train_loss.json"), "w") as f:
        json.dump(train_loss, f)
    with open(os.path.join(log_dir, "val_mAP.json"), "w") as f:
        json.dump(val_mAP, f)


def train(model, train_loader, criterion, optimizer, epoch):
    global train_loss
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    # batch_img1 and batch_img2 are both list of tensors,
    # length is args.batch_size, target is a tensor of size args.batch_size
    for batch_idx, (batch_img1, batch_img2, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time()-end)
        optimizer.zero_grad()

        batch_loss = 0  # used to record average loss of the batch
        total_loss = 0
        for i in range(len(target)):
            im1 = Variable(batch_img1[i].unsqueeze(0).cuda())
            im2 = Variable(batch_img2[i].unsqueeze(0).cuda())
            y = Variable(torch.FloatTensor([target[i]]).cuda())

            out1, out2 = model(im1, im2)
            loss = criterion(out1, out2, y)

            total_loss += loss
            batch_loss += loss.data[0]

            if (i+1) % args.real_batchsize == 0 or i == len(target)-1:
                total_loss /= len(target)
                total_loss.backward()
                total_loss = 0

        batch_loss /= len(target)
        losses.update(batch_loss, len(target))
        optimizer.step()
        # measure how much time processing this batch takes
        batch_time.update(time.time()-end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    train_loss.append(losses.avg)


def validate(model):
    model.eval()
    database_feat = []
    end = time.time()

    for i in range(len(val_set.database_imgs)):
        img = val_set.get_img(i, query=False)
        img = Variable(img.unsqueeze(0).cuda(), volatile=True)
        feat = model.forward_once(img)
        database_feat.append(feat.data)

    database_feat = torch.cat(database_feat, dim=0)
    feat_extract_time = time.time()-end

    avg_prec = AverageMeter()
    end = time.time()
    for query_id in range(len(val_set.query_imgs)):
        query_img = val_set.get_img(query_id, query=True)
        query_img = Variable(query_img.unsqueeze(0).cuda(), volatile=True)
        query_feat = model.forward_once(query_img)

        similarity = (query_feat.data*database_feat).sum(dim=1)
        _, idx = torch.sort(similarity, dim=0, descending=True)

        ap = average_precision(list(idx), val_set.gt_info[query_id])
        avg_prec.update(ap)

    query_time = time.time()-end

    print("Feature extraction time: {:.3f} ({:.3f})\t"
          "Query time: {:.3f} ({:.3f})\tmAP: {:.3f}".format(
           feat_extract_time, feat_extract_time/len(val_set.database_imgs),
           query_time, query_time/len(val_set.query_imgs), avg_prec.avg))

    return avg_prec.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    directory = "model/check_point/{}".format(args.exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, filename)
    torch.save(state, filename)

    if is_best:
        src = os.path.join(directory, "model_best.pth.tar")
        shutil.copyfile(filename, src)


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate*(0.1**(epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


if __name__ == "__main__":
    main()
