import torch
class CrossEntropy:
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()
    
    def __call__(self,preds,labels):
        if isinstance(preds,list): # for augumented data
            labels = torch.cat(labels,dim=0)
            loss = 0.0
            for i in range(len(preds)):
                loss += self.loss(preds[i],labels[i])
        else:
            loss = self.loss(preds,labels)
        return loss
