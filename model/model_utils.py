import os
import torch
import json
import torchvision  
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../model")
from model import models


def create_optim_by_name(params,optim_name:str,optim_hyper_params:dict,scheduler_name:str="",scheduler_hyper_params:dict={}):
    optimizer_config, scheduler_config = {},{}
    if optim_name == "SGD":
        optimizer = optim.SGD(params=params,**optim_hyper_params)
        optimizer_config["optimizer_name"] = "SGD"
        optimizer_config["hyper_parameters"] = optim_hyper_params
    elif optim_name == "AdamW":
        optimizer = optim.AdamW(params=params,**optim_hyper_params)
        optimizer_config["optimizer_name"] = "AdamW"
        optimizer_config["hyper_parameters"] = optim_hyper_params
    
    if scheduler_name == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,**scheduler_hyper_params)
        scheduler_config["scheduler_name"] = "MultiStepLR" 
    else:
        scheduler = None  
    setattr(optimizer,"config",optimizer_config)
    if scheduler:
        setattr(scheduler,"config",scheduler)
    return optimizer,scheduler
    
    



def train_model(net,optimizer,loss_fn,train_loader,test_loader,val_loader=None,
                n_epoch=100, n_converge = 10,device = torch.device("cpu")):
    '''
    net: deep neural network
    optimizer: torch optimizer
    loss_fn: The loss function 
             for augumented data, n_views should be passed as one of the parameters of loss_fn
    train_loader:   loader for the training set
                    imgs,labels = next(iter(train_loader)) 
                    (a) if images are augumented into n_views
                    imgs and labels a lists of length = n_views
                    imgs[i].shape = (batch_size,H,W) i=0...(n_views-1)
                    labels[i].shape = (batch_size)
                    (b) if images are not augumented
                    imgs is a batch_size*H*W tensor
                    lables is a batich_size tensor
    test_loader: loader for the test set, data should not be transfomed(except Normailze() and ToTensor())
    val_loader: loader for the validation set, data should not be transfomed(except Normailze() and ToTensor())
    n_epoch: number of epoches
    n_converge: if the validation error does not improve for n_converge steps, then stop the trainning process
    device: device 
    '''
    net.to(device)
    train_accs, val_accs = [], []
    best_epoch,best_val_acc = -1,-1.0
    # train the model
    for epoch in range(n_epoch):
        n_true,n_sample = 0,0
        for imgs,labels in train_loader:
            net.train()
            optimizer.zero_grad()
            if isinstance(imgs,list): # for augumented data
                n_views = len(imgs)
                imgs = torch.cat(imgs,dim=0)
                labels = torch.cat(labels,dim=0)
                imgs,labels = imgs.to(device),labels.to(device)
                preds = net(imgs)
                loss = loss_fn(preds,labels)
            else:
                imgs,labels = imgs.to(device),labels.to(device)
                preds = net(imgs)
                loss = loss_fn(preds,labels)
            loss.backward()
            optimizer.step()
            n_true += (torch.argmax(preds,dim=1) == labels).sum()
            n_sample += labels.size()[0]
        train_acc = n_true/n_sample
        train_accs.append(train_acc)
        print("epoch={},training accuracy is {:.3f}\n".format(epoch,train_acc))
        # model validation
        if val_loader:
            val_acc = test_model(net,val_loader,device)
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            elif epoch - best_epoch > n_converge:
                print("Early stopping at epoch = {} \n".format(epoch))
                break
    # test the model
    test_acc = test_model(net,test_loader,device)
    # return the accuracies
    acc_data = {"training_accs":[t.item() for t in train_accs],
                "val_accs":val_accs,
                "test_acc":test_acc}
    return acc_data

def test_model(net,data_loader,device):
    '''
    Test the model using untransformed datasets 
    '''
    net.eval()
    true_preds, count = 0, 0
    for imgs,labels in data_loader:
        imgs, labels = imgs[0].to(device), labels[0].to(device)
        with torch.no_grad():
            preds = net(imgs).argmax(dim=-1)
            true_preds += (preds == labels).sum().item()
            count += labels.shape[0]
    test_acc = true_preds / count
    return test_acc

class model_trainer:
    def __init__(self,net = None,optimizer = None,scheduler = None,loss = None,
                train_loader = None,test_loader = None,val_loader = None,
                logdir = "./runs",n_rec_loss = 1,n_rec_weight = 20,
                device = torch.device("cpu")):
        self.net = net 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.device = device
        # for tensorboard
        self.writer = SummaryWriter(log_dir=logdir)
        self.n_rec_loss = n_rec_loss
        self.n_rec_weight = n_rec_weight
        # To save the status of the training
        self.current_epoch = 0
        self.training_accuracy = []
        self.validation_accuracy = []
        self.training_loss = []
        self.validation_loss = []
        self.test_accuracy = -1.0
        self.test_loss = -1.0
        self.top5_arruracy = -1.0
        
        
    def continue_training(self,dir_path,n_epoch=100, n_converge = 10):
        self.load_training(dir_path)
        self.current_epoch += 1
        return self.train_model(n_epoch,n_converge)

    def train_model(self,n_epoch=100, n_converge = 10):
        '''
        net: deep neural network
        optimizer: torch optimizer
        loss_fn: The loss function 
                 for augumented data, n_views should be passed as one of the parameters of loss_fn
        train_loader:   loader for the training set
                        imgs,labels = next(iter(train_loader)) 
                        (a) if images are augumented into n_views
                        imgs and labels a lists of length = n_views
                        imgs[i].shape = (batch_size,H,W) i=0...(n_views-1)
                        labels[i].shape = (batch_size)
                        (b) if images are not augumented
                        imgs is a batch_size*H*W tensor
                        lables is a batich_size tensor
        test_loader: loader for the test set, data should not be transfomed(except Normailze() and ToTensor())
        val_loader: loader for the validation set, data should not be transfomed(except Normailze() and ToTensor())
        n_epoch: number of epoches
        n_converge: if the validation error does not improve for n_converge steps, then stop the trainning process
        device: device 
        '''
        self.net.to(self.device)
        best_epoch,best_val_acc = -1,-1.0
        # train the model
        for epoch in range(n_epoch):
            self.current_epoch += 1
            epoch_loss = 0.0
            n_iter = 0
            n_true,n_sample = 0,0
            for imgs,labels in self.train_loader:
                n_iter += 1
                self.net.train()
                self.optimizer.zero_grad()
                if isinstance(imgs,list): # for augumented data
                    n_views = len(imgs)
                    #imgs = torch.cat(imgs,dim=0)
                    #labels = torch.cat(labels,dim=0)
                    #imgs,labels = imgs.to(self.device),labels.to(self.device)
                    #preds = self.net(imgs)
                    #loss = self.loss(preds,labels)
                    imgs = [_imgs.to(self.device) for _imgs in imgs]
                    labels = [_labels.to(self.device) for _labels in labels]
                    preds = [self.net(_imgs) for _imgs in imgs]
                    loss = self.loss(preds,labels)
                else:
                    imgs,labels = imgs.to(self.device),labels.to(self.device)
                    preds = self.net(imgs)
                    loss = self.loss(preds,labels)
                loss.backward()
                self.optimizer.step()
                if isinstance(imgs,list):
                    for i in range(len(imgs)):
                        n_true += (torch.argmax(preds[i],dim=1) == labels[i]).sum()
                        n_sample += labels[i].size()[0]
                else:
                    n_true += (torch.argmax(preds,dim=1) == labels).sum()
                    n_sample += labels.size()[0]
                epoch_loss += loss.item()
            if self.scheduler:
                self.scheduler.step()
            epoch_loss /= n_iter
            # save the training accuracy and loss
            training_acc = (n_true/n_sample).item()
            print("epoch={},training accuracy is {:.3f}\n".format(self.current_epoch,training_acc))
            # model validation
            if self.val_loader:
                val_loss,val_acc = self.test(self.val_loader)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                elif epoch - best_epoch > n_converge:
                    print("Early stopping at epoch = {} \n".format(self.current_epoch))
                    break
            # record the loss and accuracy
            if epoch % self.n_rec_loss == 0:
                self.writer.add_scalar("training_accuracy",training_acc,self.current_epoch)
                self.writer.add_scalar("training_loss",loss,self.current_epoch)
                self.writer.add_scalar("validation_acc",val_acc,self.current_epoch) 
                self.training_accuracy.append(training_acc)     
                self.training_loss.append(epoch_loss)
                self.validation_accuracy.append(val_acc)
                self.validation_loss.append(val_loss)
            if epoch % self.n_rec_weight == 0:
                self.net.eval()
                count = 0 
                for name,parameter in self.net.named_parameters():
                    if count >= 16:
                        break #only record the first 16 layers
                    if 'weight' in name:
                        self.writer.add_histogram('weight:' + name,parameter.view(-1),self.current_epoch)
                        self.writer.add_histogram('grad:' + name, parameter.grad.view(-1),self.current_epoch)
                    count += 1
        self.test_loss,self.test_accuracy = self.test(self.test_loader)
        self.top5_accuracy = self.n_accuracy(self.test_loader,(5,))
        return self.training_state_dict()
    
    def test(self,data_loader):
        '''
        Test the model using untransformed datasets 
        '''
        self.net.eval()
        true_preds, count = 0, 0
        test_loss, n_iter = 0.0, 0
        for imgs,labels in data_loader:
            if isinstance(imgs,list): # for augumented data
                n_views = len(imgs)
                imgs,labels = imgs[0].to(self.device),labels[0].to(self.device) 
            else:
                imgs,labels = imgs.to(self.device),labels.to(self.device) 
            with torch.no_grad():
                preds = self.net(imgs)
                arg_preds = preds.argmax(dim=-1)
                true_preds += (arg_preds == labels).sum().item()
                count += labels.shape[0]
                test_loss += self.loss(preds,labels)
                n_iter += 1 
        test_loss /= n_iter
        test_acc = true_preds / count
        return test_loss,test_acc
    
    def n_accuracy(self,data_loader, top_k=(5,10)):
        '''
        get the top k accuracy for the net with the data set
        see https://github.com/bearpaw/pytorch-classification/blob/24f1c456f48c78133088c4eefd182ca9e6199b03/utils/eval.py#L5
        '''
        self.net.eval()
        acc = [0.0 for _ in top_k] 
        n_iter = 0
        max_k = max(top_k)
        for imgs,labels in data_loader:
            if isinstance(imgs,list): # for augumented data
                n_views = len(imgs)
                imgs = torch.cat(imgs,dim=0)
                labels = torch.cat(labels,dim=0)
            imgs,labels = imgs.to(self.device),labels.to(self.device)
            with torch.no_grad():
                _,preds_k = self.net(imgs).topk(max_k,dim = 1) # size = (batch_size*n_view,top_k)
                expanded_labels = labels.view(-1,1).expand_as(preds_k) # size = (batch_size*n_view,top_k)
                for i in range(len(top_k)):
                    k = top_k[i]
                    acc[i] += (preds_k == expanded_labels[:,:k]).float().sum(dim=1).mean()
            n_iter += 1
        acc = [val/n_iter for val in acc]
        return acc
    
    def training_state_dict(self):
        info_dict = {"training_accuracy":self.training_accuracy,
                    "training_loss":self.training_loss,
                    "validation_accuracy":self.validation_accuracy,
                    "validation_loss":self.validation_loss,
                    "test_loss":self.test_loss,
                    "test_accuracy":self.test_accuracy,
                    "top5_accuracy":self.top5_accuracy,
                    "epoch":self.current_epoch}
        return info_dict
    
    def load_optimizer(self,dir_path):
        optim_path = os.path.join(dir_path,"optimizer.tar")
        optim_config_path = os.path.join(dir_path,"optimizer_config.json")
        sched_path = os.path.join(dir_path,"scheduler.tar")
        sched_config_path = os.path.join(dir_path,"scheduler_config.json")
        with open(optim_config_path,"r") as f:
            optim_config = json.loads(f.read())
        with open(sched_config_path,"r") as f:
            sched_config = json.loads(f.read())
        optim_name = optim_config["optimizer_name"]
        scheduler_name = sched_config["scheduler_name"]
        optimizer,scheduler = create_optim_by_name(params = self.net.parameters(),
                                                   optim_name=optim_name,optim_hyper_params=optim_config["hyper_parameters"],
                                                   scheduler_name=scheduler_name,scheduler_hyper_params=sched_config["hyper_parameters"])
        optim_state_dict = torch.load(optim_path)
        optimizer.load_state_dict(optim_state_dict)
        if scheduler:
            sched_state_dict = torch.load(sched_path)
            scheduler.load_state_dict(sched_state_dict)
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def save_optimizer(self,dir_path):
        optim_path = os.path.join(dir_path,"optimizer.tar")
        optim_config_path = os.path.join(dir_path,"optimizer_config.json")
        sched_path = os.path.join(dir_path,"scheduler.tar")
        sched_config_path = os.path.join(dir_path,"scheduler_config.json")
        torch.save(self.optimizer.state_dict(),optim_path)
        with open(optim_config_path,"w") as f:
            f.write(json.dumps(self.optimizer.config, indent=4))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(),sched_path)
            with open(sched_path,"w") as f:
                f.write(json.dumps(self.scheduler.config,indent=4))
        else: # no scheduler is used
            config = {"scheduler_name":"NA","hyper_parameters":{}}
            with open(sched_config_path,"w") as f:
                f.write(json.dumps(config,indent=4))
    
    def save_training(self,dir_path):
        #save the model
        self.net.save(dir_path)
        #save the optimizer and scheduler
        self.save_optimizer(dir_path)
        #save the training information
        file_path = os.path.join(dir_path,"training_state.json")
        with open(dir_path,"w") as f:
            f.write(json.dumps(file_path,indent=4))
    
    def load_training(self,dir_path):
        # load the training information
        with open(os.path.join(dir_path,"training_state.json"),"r") as f:
            training_state = json.loads(f.read())
        for k,v in training_state.items():
            setattr(self,k,v)
        # load the network
        self.net.load(dir_path)
        # load the optimizer and the scheduler
        self.load_optimizer(dir_path)
    
        

    