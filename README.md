Implementation of [1]. The idea of this model is that if we represent input as matrices, we can reduce the amount of fully connected layer (e.g. nn.Linear) parameters many times, which potentially provide better generalization with the same amount of data. For example, nn.Linear(2000,32) has 64032 parameters, and 
NNMI ( (40,50), 32) has only 2944 parameters.

[1] P. Daniušis, Pr. Vaitkus. Neural network with matrix inputs. INFORMATICA, 2008, Vol. 19, No. 4, 477-486
