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

from model.vgg_cls_net2 import vgg16
from libs.firearm_data import FirearmCls, my_collate_cls
from libs import custom_transform

parser = argparse.ArgumentParser(description="Firearm classification")
parser.add_argument("--exp-name", type=str, default="vgg16_cls",
                    help="experiment name (default: none)")
parser.add_argument("--batch-size", type=int, default=64,
                    help="batch size for training (default: 64)")
parser.add_argument("--epochs", type=int, default=50,
                    help="number of epochs to train (default: 50)")
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
parser.add_argument("--data", type=str, default="data/firearm-train-val",
                    help="dataset root (default: data/firearm-train-val)")
parser.add_argument("--gpu-id", type=int, default=1, choices=[0, 1, 2, 3],
                    help="GPU to use (default: 1)")
parser.add_argument("--worker", type=int, default=6,
                    help="number of workers to fetch the data")
parser.add_argument("--print-freq", type=int, default=10,
                    help="training stats print frequency (default: 10)")

args = parser.parse_args()

best_acc = 0
train_loss = []
val_loss = []
val_acc = []
torch.cuda.set_device(args.gpu_id)


def main():
    global args, best_acc, val_acc, val_loss, train_loss

    model = vgg16(pretrained=True)
    # fix the parameter for first 2 blocks of vgg16
    # for param in list(model.parameters())[:8]:
    #     param.requires_grad = False
    model.cuda()
    # params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_trans = transforms.Compose([
        custom_transform.RandomResize(min_size=256, max_size=384),
        transforms.RandomHorizontalFlip(),
        custom_transform.RandomRotate(5),
        custom_transform.ColorJitter(0, 0.5, 0.5),
        transforms.ToTensor(),
        normalize
    ])
    val_trans = transforms.Compose([
        custom_transform.Resize(384),
        transforms.ToTensor(),
        normalize
    ])

    train_set = FirearmCls(root=args.data, train=True,
                           regen_train_test=True, transform=train_trans)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=my_collate_cls,
                              num_workers=args.worker, pin_memory=True)

    class_weight = train_set.class_weight
    class_weight = torch.from_numpy(class_weight).float()
    class_weight = class_weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).cuda()

    val_set = FirearmCls(root=args.data, train=False,
                         regen_train_test=False, transform=val_trans)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                            shuffle=False, collate_fn=my_collate_cls,
                            num_workers=args.worker, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        loss_train = train(model, train_loader, criterion, optimizer, epoch)
        loss_val, acc_val = validate(model, val_loader, criterion)

        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_acc.append(acc_val)

        is_best = acc_val > best_acc
        best_acc = max(acc_val, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    log_dir = os.path.join("result", args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, "expt_stat.json"), "w") as f:
        json.dump({'train_loss': train_loss,
                   'val_loss': val_loss,
                   'val_acc': val_acc}, f)


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time()-end)
        optimizer.zero_grad()

        batch_loss = 0 # used to record average loss of the batch
        total_loss = 0
        for i in range(len(target)):
            img = Variable(data[i].unsqueeze(0).cuda())
            y = Variable(torch.LongTensor([target[i]]).cuda())

            out = model(img)
            loss = criterion(out, y)

            total_loss += loss
            batch_loss += loss.data[0]

            if (i+1)%32 == 0 or i == len(target)-1:
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

    return losses.avg


def validate(model, val_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in val_loader:

        for i in range(len(target)):
            img = Variable(data[i].unsqueeze(0).cuda(), volatile=True)
            y = Variable(torch.LongTensor([target[i]]).cuda())
            out = model(img)
            loss = criterion(out, y)
            test_loss += loss.data[0]
            pred = out.data.max(1, keepdim=True)[1] #keep index
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    test_loss /= len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} '
          '({:.2f}%)\n'.format(test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    return test_loss, correct/len(val_loader.dataset)


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
    lr = args.learning_rate*(0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr


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
