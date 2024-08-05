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
    def __init__(self,n_views,batch_size,n_pow,c1,c2):
        self.n_views = n_views
        self.batch_size = batch_size
        self.n_pow = n_pow
        self.c1 = c1 # loss coefficient for repusion
        self.c2 = c2 # loss coefficient for alignment
    def __call__(self,preds,labels):
        # reshape (V*B)*O shape tensor to shape V*B*O 
        preds = torch.reshape(preds,(self.n_views,self.batch_size,preds.shape[-1]))
        # centers.shape = B*O for B ellipsoids
        centers = torch.mean(preds,dim=0)
        diff = centers[:, None, :] - centers[None, :, :]
        preds -= centers
        corr = torch.matmul(torch.permute(preds,(1,2,0)), torch.permute(preds,(1,0,2))) # size B*O*O
        # average radius for each ellipsoid
        # trace for each ellipsoids, t = torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)
        # t[i] = sum of eigenvalues for ellipsoid i, semi-axial length = sqrt(eigenvalue)
        # average radii = sqrt(sum of eigenvalues/output_dim) 
        # average radii.shape = (B,)
        radii = torch.sqrt(torch.diagonal(corr,dim1=1,dim2=2).sum(dim=1)/preds.shape[-1])
        # calculate the largest eigenvectors by power iteration method
        corr_pow = torch.stack([torch.matrix_power(corr[i], self.n_power) for i in range(corr.shape[0])])
        b0 = torch.rand(preds.shape[-1])
        eigens = torch.matmul(corr_pow,b0) # size = B*O
        eigens /= torch.norm(eigens,dim=1) 
        # loss 0: minimize the size of each ellipsoids
        ll = torch.sum(radii)
        # loss 1: repulsive loss
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        # make self distance = inf to avoid self interaction
        mask = torch.eye(dist_matrix.shape[0], dtype=torch.bool)
        dist_matrix.masked_fill_(mask, 1e12) 
        sum_radii = radii[None,:] - radii[:,None]
        pos = torch.where(dist_matrix < sum_radii)
        ll += (1.0 - (dist_matrix[pos]/sum_radii[pos])**2).sum()*self.c1
        # loss 2: alignment loss
        sim = torch.matmul(eigens,eigens.transpose(0,1))**2
        ll += (1.0 - sim[pos]).sum()*self.c2
        return ll
        


  

    
        
            

