import os
import torch
import json
import torchvision


def save_model(model,config,dir_path,overwrite = True):
    '''
    --args
    model: the torch nerual network to be saved
    config: the dict object contains all the information about the model
            e.g model_name, num_classes
    dir_path: the path where the model and the config file are saved 
    overwrite: if True, overwrite the existed model
    '''
    model_path = os.path.join(dir_path,"model.tar")
    config_path = os.path.join(dir_path,"config.json")
    if os.path.exists(model_path) and os.path.exists(config_path) and (not overwrite):
        print("File exists and overwrite = False, no IO is performed \n")
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path,exist_ok=True)
        torch.save(model.state_dict(), model_path)
        with open(config_path,"w") as f:
            f.write(json.dumps(config,indent=4))
    

def load_model(dir_path):
    model_path = os.path.join(dir_path,"model.tar")
    config_path = os.path.join(dir_path,"config.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file:{} does not exist".format(model_path)) 
    if not os.path.exists(config_path):
        raise FileNotFoundError("Config file:{} does not exist".format(config_path))
    with open(config_path,"r") as f:
        config = json.loads(f.read())
    
    if config["model_name"] == "resnet18":
        net = torchvision.models.resnet18(num_classes=config["num_classes"])
    elif config["model_name"] == "resnet34":
        net = torchvision.models.resnet34(num_classes=config["num_classes"])
    elif config["model_name"] == "resnet50":
        net = torchvision.models.resnet50(num_classes=config["num_classes"])
    elif config["model_name"] == "resnet101":
        net = torchvision.models.resnet101(num_classes=config["num_classes"])
    elif config["model_name"] == "resnet152":
        net = torchvision.models.resnet152(num_classes=config["num_classes"])
    else:
        raise NotImplementedError("Model:{} is not implemented".format(config["model_name"])) 
    net.load_state_dict(torch.load(model_path))
    return net




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
    def __init__(self,net,optimizer,loss,train_loader,test_loader,device = torch.device("cpu")):
        self.net = net 
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader = train_loader
        self.train_loader = self.train_loader
        self.device = device

    def train_model()
    