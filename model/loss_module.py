import torch
import torch.nn.functional as F
class CrossEntropy:
    def __init__(self):
        self.loss_name = "cross_entropy_loss"
        self.loss = torch.nn.CrossEntropyLoss()
    def __call__(self,preds,labels):
        loss = self.loss(preds,labels)
        return loss

class InfoNCELoss:
    def __init__(self,n_views,batch_size,tau):
        self.loss_name = "info_nce_loss"
        self.n_views = n_views
        self.batch_size = batch_size  
        self.tau = tau  
        self.hyper_parameters = {"n_views":n_views,"batch_size":batch_size,"tau":tau}
    def __call__(self,preds,labels):
        sim = F.cosine_similarity(preds[:,None,:],preds[None,:,:],dim=-1)
        mask_self = torch.eye(preds.shape[0],dtype=torch.bool,device=sim.device)
        sim.masked_fill_(mask_self,0.0)
        positive_mask = mask_self.roll(shifts=self.batch_size,dims=0)
        sim /= self.tau
        ll = torch.mean(-sim[positive_mask] + torch.logsumexp(sim,dim=-1))
        return ll

class GaussianPackingLoss:
    def __init__(self,n_views,batch_size):
        self.n_views = n_views
        self.batch_size = batch_size
    def __call__(self,preds,labels):
        pass

class EllipoidsPackingLoss:
    def __init__(self,n_views,batch_size):
        self.n_views = n_views
        self.batch_size = batch_size
    def __call__(self,preds,labels):
        # reshape (V*B)*O shape tensor to shape V*B*O 
        preds = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        centers = torch.mean(preds,dim=0)
        v = preds - centers
        corr = torch.matmul(torch.permute(v,(1,2,0)), torch.permute(v,(1,0,2))) # size B*D*D


    
        
            

