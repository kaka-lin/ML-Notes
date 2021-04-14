import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """Focal Loss

        focal loss: -α(1-pt)**gamma * logpt
        :param alpha: the balance factor for this criterion. (class weight)
                      0 <= alpha <= 1,
                      positive samples: alpha (class 1)
                      negative samples: 1-alpha (class -1)
        :param gamma: reduces the relative loss for well-classiﬁed examples (pt > .5),
                      putting more focus on hard, misclassiﬁed examples.
        :param size_average: By default, the losses are averaged over observations for each minibatch.
                             However, if the field size_average is set to False,
                             the losses are instead summed for each minibatch.
        """
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, preds, labels):
        """
        :param preds: pred class.
        :param labels: ground truth.
        """
        if preds.dim() > 2:
            preds = preds.view(preds.size(0), preds.size(1), -1) # N, C, H, W -> N, C, H*W
            preds = preds.transpose(1, 2) # N, C, H*W -> N, H*W, C
            preds = preds.contiguous().view(-1, preds.size(2)) # N, H*W, C -> N*H*W, C

        labels = labels.view(-1, 1)
        logpt = F.log_softmax(preds)
        # Gathers values along an axis specified by dim.
        logpt = logpt.gather(1, labels)
        logpt = logpt.view(-1)
        pt = Variable(logpt.exp())

        if self.alpha is not None:
            if self.alpha.type() != preds.data.type():
                self.alpha = self.alpha.type_as(preds)
            at = self.alpha.gather(0, labels.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()
