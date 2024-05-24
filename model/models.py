import os
import torch
import json
import torchvision

class BaseNetwork(torch.nn.Module):
    '''
    any subclass must have 1) self.name initialized
                           2) self.net object created
                           3) self.hyper_parameters (dictionary) created
                        in the __init__() function
    '''
    def __init__(self):
        super().__init__()
        self.model_name = type(self).__name__         
        self.net = None
        self.hyper_parameters = None

    def save(self,dir_path,overwrite = True):
        model_path = os.path.join(dir_path,"model.tar")
        config_path = os.path.join(dir_path,"model_config.json")
        if os.path.exists(model_path) and os.path.exists(config_path) and (not overwrite):
            print("File exists and overwrite = False, no IO is performed \n")
        else:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path,exist_ok=True)
            config = {}
            config["model_name"] = self.model_name
            config["hyper_parameters"] = self.hyper_parameters
            torch.save(self.state_dict(), model_path)
            with open(config_path,"w") as f:
                f.write(json.dumps(config,indent=4))
    
    def load(self,dir_path):
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
        if config["hyper_parameters"]:
            for k,v in config["hyper_parameters"].items():
                setattr(self,k,v)
        self.load_state_dict(torch.load(model_path))
        return config

class Resnet34(BaseNetwork):
    def __init__(self,num_classes:int):
        super().__init__()
        self.model_name = "Resnet34"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"num_classes":num_classes}
        for k,v in self.hyper_parameters.items():
            setattr(self,k,v)
        # construct the model
        self.net = torchvision.models.resnet34(num_classes = num_classes)
    
    def forward(self,x):
        return self.net(x)

class Resnet18(BaseNetwork):
    def __init__(self,num_classes:int):
        super().__init__()
        self.model_name = "Resnet18"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"num_classes":num_classes}
        for k,v in self.hyper_parameters.items():
            setattr(self,k,v)
        # construct the model
        self.net = torchvision.models.resnet18(num_classes = num_classes)
    
    def forward(self,x):
        return self.net(x)

class SimpleCLRNet(BaseNetwork):
    def __init__(self,embedded_dim:int,resnet_type:str="resnet18"):
        super().__init__()
        self.model_name = "SimpleCLRNet"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"embedded_dim":embedded_dim,"resnet_type":resnet_type}
        for k,v in self.hyper_parameters.items():
            setattr(self,k,v)
        if resnet_type == "resnet18":
            self.net = torchvision.models.resnet18(num_classes = 4*embedded_dim)
        elif resnet_type == "resnet34":
            self.net = torchvision.models.resnet34(num_classes = 4*embedded_dim)
        
        self.net.fc = torch.nn.Sequential(
                        self.net.fc,
                        torch.nn.ReLU(),
                        torch.nn.Linear(4*embedded_dim,embedded_dim)
                    )
    
    def remove_projection_layer(self):
        self.net.fc = torch.nn.Identity()
    
    def add_linear_layer(self,out_dim):
        self.net.fc = torch.nn.Sequential(
            self.net.fc,
            torch.nn.Linear(in_features=4*self.embedded_dim,out_features= out_dim)
        )

    def forward(self,x):
        return self.net(x)

class LinearNet(BaseNetwork):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.model_name = "LinearNet"
        # this is a sloppy way of recording hyperperameters
        self.hyper_parameters = {"in_dim":in_dim,"out_dim":out_dim}
        for k,v in self.hyper_parameters.items():
            setattr(self,k,v)
        self.net = torch.nn.Linear(in_features=in_dim,out_features=out_dim)
    
    def forward(self,x):
        return self.net(x)








    

        

        