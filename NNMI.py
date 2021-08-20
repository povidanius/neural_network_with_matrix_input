import torch
import torch.nn as nn


# P. Daniušis, Pr. Vaitkus. Neural network with matrix inputs. INFORMATICA, 2008, Vol. 19, No. 4, 477-486

class NNMI(nn.Module):
    def __init__(self, size, num_features):
        super(NNMI, self).__init__()
        self.rows = size[0]
        self.cols = size[1]
        self.conv1 = nn.Conv2d(1,num_features, kernel_size=(1,self.cols))
        self.conv2 = nn.Conv2d(num_features,num_features, kernel_size=(self.rows,1),groups=num_features)
        self.relu = nn.ReLU()
        self.type = 'NNMI'

    def forward(self,x):
        x = x.view(-1,1,self.rows,self.cols)        
        x = self.conv2(self.conv1(x)).squeeze()
        x = self.relu(x)
        #print(x.shape)
        return x
