import os
import torch
import json
import torchvision

class BaseNetwork(torch.nn.modules):
    '''
    any subclass must have 1) self.name initialized
                           2) self.net object created
                           3) self.hyper_parameters (dictionary) created
                        in the __init__() function
    '''
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.model_name = type(self).__name__         
        self.net = None
        self.hyper_parameters = {}

    def save_model(self,dir_path,overwrite = True):
        model_path = os.path.join(dir_path,"model.tar")
        config_path = os.path.join(dir_path,"model_config.json")
        if os.path.exists(model_path) and os.path.exists(config_path) and (not overwrite):
            print("File exists and overwrite = False, no IO is performed \n")
        else:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path,exist_ok=True)
            config = {}
            config["model_name"] = self.model_name
            config.update(self.hyper_parameters)
            torch.save(self.state_dict(), model_path)
            with open(config_path,"w") as f:
                f.write(json.dumps(config,indent=4))
    
    def load_model(self,dir_path):
        model_path = os.path.join(dir_path,"model.tar")
        config_path = os.path.join(dir_path,"model_config.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file:{} does not exist".format(model_path)) 
        if not os.path.exists(config_path):
            raise FileNotFoundError("Config file:{} does not exist".format(config_path))
        with open(config_path,"r") as f:
            config = json.loads(f.read())
        if config["model_name"] != self.model_name:
            raise TypeError("cannot load {} into {} object".format(config["model_name"],self.model_name))
        # update hyper_peremeters 
        if self.hyper_parameters:
            for k,v in self.hyper_parameters:
                setattr(self,k,v)
        self.net.load_state_dict(torch.load(model_path))

class Resnet34(BaseNetwork):
    def __init__(self,num_classes:int):
        super.__init__()
        self.model_name = "Resnet34_num_class" + str(num_classes)
        self.hyper_parameters = {}
        # construct the model
        self.net = torchvision.models.resnet34(num_classes = num_classes)
    
    def forward(self,x):
        return self.net(x)



    

        

        