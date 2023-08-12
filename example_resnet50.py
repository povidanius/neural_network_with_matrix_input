import torch
import torch.nn as nn
from torchvision.models import resnet50
from NNMI import NNMI, get_num_parameters

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Resnet50_NNMI(nn.Module):
    def __init__(self):
        super(Resnet50_NNMI, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = Identity()
        self.nnmi = NNMI( (64,32), 1000)

    def forward(self, x):
        embedding = self.resnet(x)       
        out = self.nnmi(embedding.view(-1,64,32))
        return out 


if __name__ == "__main__":
    x = torch.randn((16, 3, 224, 224))      
    model = Resnet50_NNMI()
    output = model(x)
    print("Num parameters: {}".format(get_num_parameters(model.nnmi)))
    breakpoint()


