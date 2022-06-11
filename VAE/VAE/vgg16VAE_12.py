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

class Vgg16VAE(nn.Module):


    def __init__(self,pool=None,device=None):
        super(Vgg16VAE,self).__init__()
        self.pretrained_model = models.vgg16(pretrained=True)
        self.device=None
        if(device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.FCM = torch.nn.Linear(64*8*8,12).to(self.device)
        self.FCV = torch.nn.Linear(64*8*8,12).to(self.device)


        self.encoder = nn.Sequential(*list(self.pretrained_model.features.children())[:-1],
                                            nn.AvgPool2d(2, stride=2),
                                            nn.Conv2d(512,64,kernel_size=3,stride=1,padding=1)
                                            #nn.PReLU(),
                                            #nn.Flatten(), # batch_size,2048
                                            #nn.Linear(2048,256),
                                            #nn.PReLU()
                                            )
        self.decoder =  torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,64,3,stride=1),        #   (64,6,6) 6
            torch.nn.ReLU(), #(6,5)
            torch.nn.ConvTranspose2d(64,512,3,stride=2,padding=1,output_padding=1),        #   (512,13,13) 14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512,256,3,stride=2,padding=1,output_padding=1),        #   (256,27,27) 30
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),        #   (128,55,55) 62
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),        #   (64,111,111) 127
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),        #   (3,231,231) 254
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32,3,3,stride=2,padding=1,output_padding=1)        #   (3,231,231) 254
        )

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
        v = self.FCV(feature)
        feature = self.reparametrize(mu,v)
        feature = feature.view(-1,3,2,2)
        out = self.decoder(feature)
        return out,mu,v





