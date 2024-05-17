import torch
class CrossEntropy:
    def __init__(self,n_views):
        self.loss = torch.nn.CrossEntropyLoss()
        self.n_veiws = n_views
    def __call__(self,preds,labels):
        if isinstance(preds,list): # for augumented data
            loss = 0.0
            for i in range(len(preds)):
                loss += self.loss(preds[i],labels[i])
        else:
            loss = self.loss(preds,labels)
        return loss
