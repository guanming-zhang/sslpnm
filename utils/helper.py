import torch
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def set_random_seed(s:int=137):
    torch.manual_seed(s) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device torch.device("cpu")
    return device

def show_images(imgs,nrow,ncol,titles = None):
    '''
    --args 
    imgs: a list of images(PIL or torch.tensor or numpy.ndarray)
    nrow: the number of rows
    ncol: the number of columns
    titles: the tile of each subimages
    note that the size an image represented by PIL or ndarray is (W*H*C),
              but for tensor it is (C*W*H)
    --returns
    fig and axes
    '''
    fig,axes = plt.subplots(nrow,ncol)
    for i in range(len(imgs)):
        row  = i // ncol
        col = i % ncol
        if titles:
            axes[row,col].set_title(titles[i])
        if isinstance(imgs[i],Image.Image):
            img = np.array(imgs[i])
        elif torch.is_tensor(imgs[i]):
            img = imgs[i].cpu().detach()
            img = img.permute((1,2,0)).numpy()
        elif isinstance(imgs[i], np.ndarray):
            img = imgs[i]
        else:
            raise TypeError("each image must be an PIL or torch.tensor or numpy.ndarray")
        axes[row,col].imshow(img)
        axes[row,col].set_axis_off()
    return fig,axes

#####################################
# image augumentation
#####################################
class AugumentationTrans(object):
    '''
    for image augumentation
    '''
    def __init__(self, my_transforms, n_views=2):
        '''
        --args:
        my_transforms: torchvison.transforms object that transforms the image
        n_views: the number of transfomed images augmented for each orginal image
        '''
        self._transforms = my_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self._transforms(x) for i in range(self.n_views)]

class WrappedDataset(Dataset):
    '''
    This class is designed to apply diffent transforms to subdatasets
    subdatasets are not allowed to have different transforms by default
    By wrapping subdatasets to WrappedDataset, this problem is solved
    e.g 
    _train_set, _val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_set = WrappedDataset(_train_set,transforms.RandomHorizontalFlip(), n_views=3)
    val_set = WrappedDataset(_val_set,transforms.ToTensor())
    '''
    def __init__(self, dataset, transform=None, n_views = 1):
        self.dataset = dataset
        self.transform = transform
        self.n_views = n_views
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = [self.transform(x) for i in range(self.n_views)]
            y = [y for i in range(self.n_views)]
        return x, y
        
    def __len__(self):
        return len(self.dataset)

#####################################
# For CIFAR10 dataset
#####################################   
def get_cifar10_classes():
    labels = ["airplane","automobile","bird","cat",
              "deer","dog","frog","horse","ship","truck"]
    return labels