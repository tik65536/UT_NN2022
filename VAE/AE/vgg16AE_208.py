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

        self.FCM = torch.nn.Linear(64*8*8,208).to(self.device)



        self.encoder = nn.Sequential(*list(self.pretrained_model.features.children())[:-1],
                                            nn.AvgPool2d(2, stride=2),
                                            nn.Conv2d(512,64,kernel_size=3,stride=1,padding=1)
                                            #nn.PReLU(),
                                            #nn.Flatten(), # batch_size,2048
                                            #nn.Linear(2048,256),
                                            #nn.PReLU()
                                            )

        self.decoder = nn.Sequential(
                        # assume the input is 8*8*4
                        nn.Conv2d(13,64,kernel_size=3,stride=1,padding=1),  #decode 1
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),  #decode 2
                        nn.ReLU(inplace=True),

                        nn.Upsample(scale_factor=4,mode='nearest'), # 16 ,32
                        nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1), #decode 3
                        nn.ReLU(inplace=True),

                        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1), #decode 4
                        nn.ReLU(inplace=True),

                        nn.Upsample(scale_factor=2,mode='nearest'),# 128 , 64
                        nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),   #decode 5
                        nn.ReLU(inplace=True),

                        nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),   #decode 6
                        nn.ReLU(inplace=True),

                        nn.Upsample(scale_factor=2,mode='nearest'), # 128 ,128
                        nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),   #decode 7
                        nn.ReLU(inplace=True),

                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),   #decode 8
                        nn.ReLU(inplace=True),

                        nn.Upsample(scale_factor=2,mode='nearest'), # 128 ,256
                        nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),   #decode 9
                        nn.ReLU(inplace=True),

                        nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),   #decode 7
                        nn.ReLU(inplace=True),

                        nn.Upsample(scale_factor=2,mode='nearest'), # 32 ,128 ,128
                        nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),   #decode 8
                        nn.ReLU(inplace=True),

                        nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1)   #decode 9
                        )

        #self.decoder.apply(self.weight_init)
        del self.pretrained_model
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

        feature = mu.view(-1,13,4,4)
        out = self.decoder(feature)
        return out,mu





