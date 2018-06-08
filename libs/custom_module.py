import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
    """
    input two tensor input x1 and x2 and a label tensor y indicating whether
    corresponding sample in x1 and x2 are similar or not.

    shape of x1 and x2: N*D, N is batchsize, D is feature dimension
    shape of y: N
    """
    def __init__(self, margin, weight=1.0, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        self.eps = eps

    def forward(self, x1, x2, y):
        diff = torch.abs(x1-x2)
        dist_sq = torch.pow(diff+self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        mdist = torch.clamp(self.margin-dist, min=0.0)

        loss_pos = y*dist_sq
        loss_neg = (1-y)*(mdist.pow(2))
        loss = (loss_pos + self.weight*loss_neg)*0.5

        return torch.mean(loss)


class SoftContrastiveLoss(nn.Module):
    def __init__(self, margin1, margin2, eps=1e-6):
        super(SoftContrastiveLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, y):
        diff = torch.abs(x1-x2)
        dist_sq = torch.pow(diff+self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        mdist_pos = torch.clamp(dist-self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2-dist, min=0.0)

        # print(y.data.type(), mdist_pos.data.type(), mdist_neg.data.type())
        loss_pos = y*(mdist_pos.pow(2))
        loss_neg = (1-y)*(mdist_neg.pow(2))

        loss = (loss_pos + loss_neg)*0.5

        return torch.mean(loss)


# without using square
class SoftContrastiveLoss2(nn.Module):
    def __init__(self, margin1, margin2, eps=1e-6):
        super(SoftContrastiveLoss2, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.eps = eps

    def forward(self, x1, x2, y):
        diff = torch.abs(x1-x2)
        dist_sq = torch.pow(diff+self.eps, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        mdist_pos = torch.clamp(dist-self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2-dist, min=0.0)

        # print(y.data.type(), mdist_pos.data.type(), mdist_neg.data.type())
        loss_pos = y*mdist_pos
        loss_neg = (1-y)*mdist_neg

        loss = (loss_pos + loss_neg)*0.5

        return torch.mean(loss)


if __name__ == "__main__":
    from torch.autograd import Variable
    import torch.nn.functional as F
    #
    # num = 4
    # f1 = Variable(torch.rand(num, 64))
    # f2 = Variable(torch.rand(num, 64))
    #
    # f1 = F.normalize(f1, p=2, dim=1)
    # f2 = F.normalize(f2, p=2, dim=1)
    #
    # y = Variable(torch.rand(num))
    #
    # criterion = SoftContrastiveLoss(margin1=0.5, margin2=0.8)
    # dist, loss = criterion(f1, f2, y)
    #
    # diff = torch.abs(f1-f2)
    # dist_sq = torch.pow(diff+1e-6, 2).sum(dim=1)
    # dist_m = torch.sqrt(dist_sq)
    #
    # mdist_pos = torch.clamp(dist_m - 0.5, min=0.0)
    # mdist_neg = torch.clamp(0.8-dist_m, min=0.0)
    #
    # loss_pos = y*(mdist_pos.pow(2))
    # loss_neg = (1-y)*(mdist_neg.pow(2))
    #
    # loss_m = torch.mean((loss_pos+loss_neg)*0.5)

    # diff = torch.abs(f1 - f2)
    # dist_sq = torch.pow(diff + 1e-6, 2).sum(dim=1)
    # dist_m = torch.sqrt(dist_sq)
    #
    # mdist_pos = torch.clamp(dist_m - 0.5, min=0.0)
    # mdist_neg = torch.clamp(0.8 - dist_m, min=0.0)
    #
    # loss_pos = y * (mdist_pos.pow(2))
    # loss_neg = (1 - y) * (mdist_neg.pow(2))
    #
    # loss_m = torch.mean((loss_pos + loss_neg) * 0.5)

    # print(loss, loss_m)
    # print(dist, dist_m)