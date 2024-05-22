import torch
import torch.nn.functional as F
class CrossEntropy:
    def __init__(self,n_views):
        self.loss = torch.nn.CrossEntropyLoss()
        self.n_veiws = n_views
    def __call__(self,preds,labels):
        loss = self.loss(preds,labels)
        return loss

class InfoNCELoss:
    def __init__(self,n_views,batch_size):
        self.n_views = n_views
        self.batch_size = batch_size    
    def __call__(self,preds,labels):
        sim = F.cosine_similarity(preds[:,None,:],labels[None,:,:],dim=-1)
        sim_sum = sim.sum()
        sim = sim.reshape(self.n_views,self.batch_size,self.n_views,self.batch_size)
        for i in range(self.n_views):
            for j in range(self.batch_size):
                denorminator = torch.sum(sim[i,k,:,:]) 
                for k in range(j+1,self.batch_size):
                    numerator = sim[i,j,i,k]
            
            

