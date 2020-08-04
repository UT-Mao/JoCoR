import numpy as np
import torch

from config import opt
from data.cifar import CIFAR10
from model import CNN, JoCor_loss, loss_coteaching, JoCor_loss_backward_only

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def train_JoCoR(train_loader,epoch, cnn1, cnn2, optimizer1, optimizer2, rate_schedule):
    correct_1 = 0
    correct_2 = 0
    
    loss_t_total = 0
    total_instance = 0 
    
    loss_instance = len(train_loader)

    for i,(imgs, labels, index) in enumerate(train_loader):
        y1 = cnn1(imgs.cuda())
        y2 = cnn2(imgs.cuda())

        target = labels.long().cuda()
        loss_t = JoCor_loss(y1, y2, target, rate_schedule[epoch], 0.85)

        correct_1 += sum(y1.argmax(axis = 1) ==  target)
        correct_2 += sum(y2.argmax(axis = 1) ==  target)
        total_instance += imgs.shape[0]

        loss_t_total += loss_t.item()

        optimizer2.zero_grad()
        optimizer1.zero_grad()
        
        loss_t.backward()
        
        optimizer1.step()
        optimizer2.step()
    
    return loss_t_total / loss_instance , correct_1.long().item() / total_instance , correct_2.long().item() / total_instance


def train_co_teaching(train_loader,epoch, cnn1, cnn2, optimizer1, optimizer2, rate_schedule):
    correct_1 = 0
    correct_2 = 0
    
    loss_1_total = 0
    loss_2_total = 0
    total_instance = 0 

    loss_instance = len(train_loader)

    for i,(imgs, labels, index) in enumerate(train_loader):
        y1 = cnn1(imgs.cuda())
        y2 = cnn2(imgs.cuda())

        target = labels.long().cuda()
        loss_1, loss_2 = loss_coteaching(y1, y2, target, rate_schedule[epoch], 0.85)

        correct_1 += sum(y1.argmax(axis = 1) ==  target)
        correct_2 += sum(y2.argmax(axis = 1) ==  target)
        total_instance += imgs.shape[0]

        loss_1_total += loss_1.item()
        loss_2_total += loss_2.item() 

        optimizer2.zero_grad()
        optimizer1.zero_grad()
        
        loss_1.backward()
        loss_2.backward()

        optimizer1.step()
        optimizer2.step()
    
    return loss_1_total / loss_instance, loss_2_total / loss_instance, correct_1.long().item() / total_instance , correct_2.long().item() / total_instance



def test(test_loader, cnn1, cnn2):
    correct_1 = 0
    correct_2 = 0
    total = 0

    for i,(imgs, labels, index) in enumerate(test_loader):
        y1 = cnn1(imgs.cuda())
        y2 = cnn2(imgs.cuda())
        target = labels.long().cuda()
        total += imgs.shape[0]
        correct_1 += sum(y1.argmax(axis = 1) ==  target)
        correct_2 += sum(y2.argmax(axis = 1) ==  target)
        
    return correct_1.long().item()/ total, correct_2.long().item()/ total

def set_schedule(opt):
    alpha_plan = [opt.learning_rate] * opt.n_epoch
    beta_plan = [opt.mom1] * opt.n_epoch
    for i in range(opt.epoch_decay_start, opt.n_epoch):
        alpha_plan[i] = float(opt.n_epoch - i) / (opt.n_epoch - opt.epoch_decay_start) * opt.learning_rate
        beta_plan[i] = opt.mom2
    rate_schedule = np.ones(opt.n_epoch)* opt.forget_rate
    rate_schedule[:opt.num_gradual] = np.linspace(0, opt.forget_rate**opt.exponent, opt.num_gradual)
    return alpha_plan, beta_plan, rate_schedule

def adjust_learning_rate(optimizer, epoch, alpha_plan, beta_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta_plan[epoch], 0.999)


data = CIFAR10(noise_rate = opt.noise_rate)
train_loader = torch.utils.data.DataLoader(dataset=data,batch_size=128,drop_last=True,shuffle=True)

test_data = CIFAR10(train=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=128)

cnn1 = CNN().cuda()
optimizer1 = torch.optim.Adam(cnn1.parameters(), lr= opt.learning_rate)

cnn2 = CNN().cuda()
optimizer2 = torch.optim.Adam(cnn2.parameters(), lr= opt.learning_rate)

if opt.load:
    cnn1.load_state_dict(torch.load(opt.cnn1_path))
    cnn2.load_state_dict(torch.load(opt.cnn2_path))

alpha_plan, beta_plan, rate_schedule = set_schedule(opt) 

for epoch in range(opt.n_epoch):
    cnn1.train()
    adjust_learning_rate(optimizer1, epoch, alpha_plan, beta_plan)
    cnn2.train()
    adjust_learning_rate(optimizer2, epoch, alpha_plan, beta_plan)
    
    if opt.loss in ['JoCoR', 'JoCoR_backward_only']:
        loss_t_total, correct_1, correct_2 = train_JoCoR(train_loader,epoch, cnn1, cnn2, optimizer1, optimizer2, rate_schedule)
    else:
        loss_1_total, loss_2_total, correct_1, correct_2 = train_co_teaching(train_loader,epoch, cnn1, cnn2, optimizer1, optimizer2, rate_schedule,)

    cnn1.eval()
    cnn2.eval()
    acc1_test, acc2_test = test(test_loader, cnn1, cnn2)

    if opt.loss in ['JoCoR', 'JoCoR_backward_only']:
        print('epoch',epoch,'|loss:' '%.4f' % loss_t_total ,'|acc1:''%.3f' % correct_1 , '|acc2:''%.3f' % correct_2, '|acc1_t:''%.3f' % acc1_test , '|acc2_t:''%.3f' % acc2_test)
    else:
        print('epoch',epoch,'|loss1:' '%.4f' % loss_1_total ,'|loss1:' '%.4f' % loss_2_total, '|acc1:''%.3f' % correct_1 , '|acc2:''%.3f' % correct_2, '|acc1_t:''%.3f' % acc1_test , '|acc2_t:''%.3f' % acc2_test)
if opt.save:
    torch.save(cnn1.state_dict(), opt.save_path_1)
    torch.save(cnn2.state_dict(), opt.save_path_2)
