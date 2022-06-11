#!/usr/bin/python3

import torch
from torchvision import models
import torch.nn as nn

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

class Vgg16AE(nn.Module):
    def __init__(self,pool=None,device=None):
        super(Vgg16AE,self).__init__()
        self.pretrained_model = models.vgg16(pretrained=True)
        self.device=None
        if(device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.FCM = torch.nn.Linear(64*8*8,4).to(self.device)

        self.encoder = nn.Sequential(*list(self.pretrained_model.features.children())[:-1],
                                            nn.AvgPool2d(2, stride=2),
                                            nn.Conv2d(512,64,kernel_size=3,stride=1,padding=1)
                                            #nn.PReLU()
                                            )
        self.decoder =  torch.nn.Sequential(
            # input(1,2,2)
            torch.nn.ConvTranspose2d(1,64,2,stride=1),        #   (64,3,3) 6
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,512,3,stride=2),        #   (512,7,7) 14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512,256,3,stride=2),        #   (256,15,15) 30
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256,128,3,stride=2),        #   (128,31,31) 62
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128,64,3,stride=2),        #   (64,63,63) 127
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,16,3,stride=2),        #   (3,127,127) 254
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16,3,4,stride=2)        #   (3,256,256) 254

        ).to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(log_var,device=self.device)
        #z= z.type_as(mu)
        return mu + sigma*z
    def forward(self,img):
        feature = self.encoder(img)
        feature = feature.view(-1,64*8*8)
        mu = self.FCM(feature)
        feature = mu.view(-1,1,2,2)
        out = self.decoder(feature)
        return out,mu



