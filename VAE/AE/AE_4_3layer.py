import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight,mean=0.0, std=0.00001)
        #m.bias.data.fill_(0)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight,mean=0.0, std=0.00001)
        #m.bias.data.fill_(0)
    if isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight,mean=0.0, std=0.00001)
        #m.bias.data.fill_(0)

class AE(torch.nn.Module):
    def __init__(self,ek_size=3,dk_size=3,channel=3,pool='avg',device=None):
        super(AE, self).__init__()
        if(device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.epaddingsize = ek_size//2
        self.dpaddingsize = dk_size//2
        if(ek_size%2==0):
            self.epaddingsize = ek_size//2-1

        if(dk_size%2==0):
            self.dpaddingsize = dk_size//2-1
        # input dim 3*102*188 = 57528
        # H = [ (HIn + 2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/s ] + 1
        #   =    [( 102 + 0 - 1x(2) -1 )/2] + 1
        # W = [ (WIn +2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/s ] + 1
        #   = [ ( 188 + 0 - 1x(2) -1 ) /2 ] + 1
        # ConvTranspose
        # Out =(In−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        #     = 4x2 - 0 + 2 + 0 + 1 = 11
        #     = 9x2 - 0 + 2 + 0 + 1 = 21

        #self.FC1 = torch.nn.Linear(512*2*4,3000).to(self.device)
        self.FCM = torch.nn.Linear(512*4*4,4).to(self.device)

        if(pool=='avg'):
            self.encoder =  torch.nn.Sequential(
                torch.nn.Conv2d(channel, 64, ek_size, stride=1,padding=self.epaddingsize),  #  C=64,H=102,W=188
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(4, stride=4),                       #  C=64,H=50,W=93
                torch.nn.Conv2d(64, 256, 3,stride=1,padding=1),  #  C=128,H=50,W=93
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(4, stride=4),                                             #  C=128,H=24,W=46
                torch.nn.Conv2d(256, 512, 3,stride=1,padding=1),   #  C=256,H=24,W=46
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(4, stride=4)                                        #  C=256,H=11,W=22
                                                                    #  C=512,H=5,W=10 = 25600 - mu C[0:256] , v C[256:512]
            ).to(self.device)
        else:
            self.encoder =  torch.nn.Sequential(
                torch.nn.Conv2d(channel, 64, ek_size, stride=1,padding=self.epaddingsize),  #  C=64,H=102,W=188
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(4, stride=4),                       #  C=64,H=50,W=93
                torch.nn.Conv2d(64, 256, 3,stride=1,padding=1),  #  C=128,H=50,W=93
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(4, stride=4),                                             #  C=128,H=24,W=46
                torch.nn.Conv2d(256, 512, 3,stride=1,padding=1),   #  C=256,H=24,W=46
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(4, stride=4)                   
                #self.activation
                                                         #  C=512,H=5,W=10 = 25600 - mu C[0:256] , v C[256:512]
            ).to(self.device)

        self.decoder =  torch.nn.Sequential(
            # input(1,2,2)
            torch.nn.ConvTranspose2d(1,64,3,stride=1),        #   (64,3,3) 6
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,512,4,stride=4),        #   (512,7,7) 14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512,64,4,stride=4),        #   (256,15,15) 30
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,3,4,stride=4)       #   (128,31,31) 62
         
        ).to(self.device)
        #self.encoder.apply(weights_init)
        #self.decoder.apply(weights_init)
        #torch.nn.init.normal_(self.FCM.weight,mean=0.0, std=0.00001)
        #torch.nn.init.normal_(self.FCV.weight,mean=0.0, std=0.00001)

        #self.FC1.bias.data.fill_(0)
        #self.FCM.bias.data.fill_(0)
        #self.FCV.bias.data.fill_(0)

    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(log_var,device=self.device)
        #z= z.type_as(mu)
        return mu + sigma*z
    def forward(self, face):
        facefeature = self.encoder(face)
        facefeature = facefeature.view(-1,512*4*4)
        #facefeature = self.faceFC1(facefeature)
        #facefeature = self.relu(facefeature)
        #print(f'sampling {torch.max(sampling)}')
        mu = self.FCM(facefeature)

        faceout = mu.view(-1,1,2,2)
        faceout = self.decoder(faceout)



        return faceout,mu#,scrambleout,scramblemu,scramblev
