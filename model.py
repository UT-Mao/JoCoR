import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25):
        self.dropout_rate = dropout_rate
        super(CNN, self).__init__()
        
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)

        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x):
        x=self.c1(x)
        x=F.leaky_relu(self.bn1(x), negative_slope=0.01)
        x=self.c2(x)
        x=F.leaky_relu(self.bn2(x), negative_slope=0.01)
        x=self.c3(x)
        x=F.leaky_relu(self.bn3(x), negative_slope=0.01)
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        x=F.dropout2d(x, p=self.dropout_rate)

        x=self.c4(x)
        x=F.leaky_relu(self.bn4(x), negative_slope=0.01)
        x=self.c5(x)
        x=F.leaky_relu(self.bn5(x), negative_slope=0.01)
        x=self.c6(x)
        x=F.leaky_relu(self.bn6(x), negative_slope=0.01)
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        x=F.dropout2d(x, p=self.dropout_rate)

        x=self.c7(x)
        x=F.leaky_relu(self.bn7(x), negative_slope=0.01)
        x=self.c8(x)
        x=F.leaky_relu(self.bn8(x), negative_slope=0.01)
        x=self.c9(x)
        x=F.leaky_relu(self.bn9(x), negative_slope=0.01)
        x=F.avg_pool2d(x, kernel_size=x.data.shape[2])

        x = x.view(x.size(0), x.size(1))
        x=self.l_c1(x)

        return x

def loss_coteaching(y_1, y_2, t, forget_rate):

    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

def JoCor_loss(y_1, y_2, t, forget_rate, alpha):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    KLloss = nn.KLDivLoss(reduction ='none')
    
    y_1 = torch.softmax(y_1, dim = 1)
    y_2 = torch.softmax(y_2, dim = 1)
    
    loss_3 = KLloss(y_1.log(), y_2).sum(1) + KLloss(y_2.log(), y_1).sum(1)
    loss_t = (1 - alpha) * (loss_1 + loss_2) + alpha * loss_3
    
    ind_t_sorted = np.argsort(loss_t.data.cpu()).cuda()
    loss_t_sorted = loss_t[ind_t_sorted]
    
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_t_sorted))
    ind_t_update=ind_t_sorted[:num_remember]
    
    loss_t_update = loss_t[ind_t_update]
    
    return torch.mean(loss_t_update), num_remember

def JoCor_loss_backward_only(y_1, y_2, t, forget_rate, alpha):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    KLloss = nn.KLDivLoss(reduction ='none')
    
    y_1 = torch.softmax(y_1, dim = 1)
    y_2 = torch.softmax(y_2, dim = 1)
    
    loss_3 = KLloss(y_1.log(), y_2).sum(1) + KLloss(y_2.log(), y_1).sum(1)
    
    # here is different from the original paper
    # Select clean sample without the KL_loss

    loss_t = (1 - alpha) * (loss_1 + loss_2)# + alpha * loss_3
    
    ind_t_sorted = np.argsort(loss_t.data.cpu()).cuda()
    
    loss_t = (1 - alpha) * (loss_1 + loss_2) + alpha * loss_3
    loss_t_sorted = loss_t[ind_t_sorted]
    
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_t_sorted))
    ind_t_update=ind_t_sorted[:num_remember]
    
    loss_t_update = loss_t[ind_t_update]
    
    return torch.mean(loss_t_update), num_remember







