import torch
class CrossEntropy:
    def __init__(self,n_views):
        self.loss = torch.nn.CrossEntropyLoss()
        self.n_veiws = n_views
    def __call__(self,preds,labels):
        if isinstance(preds,list): # for augumented data
            #imgs = torch.cat(imgs,dim=0)
            #labels = torch.cat(labels,dim=0)
            #imgs,labels = imgs.to(self.device),labels.to(self.device)
            #preds = self.net(imgs)
            #loss = self.loss(preds,labels)

            preds = torch.cat(preds,dim=0)
            labels = torch.cat(labels,dim=0)
            loss = self.loss(preds,labels)


            #loss = 0.0
            #for i in range(len(preds)):
            #    loss += self.loss(preds[i],labels[i])
        else:
            loss = self.loss(preds,labels)
        return loss
