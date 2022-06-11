#!/usr/bin/python3
from os import listdir
import sys
sys.path.insert(0,'../')
from os.path import isfile, join
from imagedataset import ImageDatasetPath
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
from npeet import entropy_estimators as ee
from torchsummary import summary
import numpy as np
import os
import time
import datetime
from scipy.stats import kstat
import pymp
import importlib
pymp.config.nested=True

_round = sys.argv[1]
_pool = sys.argv[2]
_bsize = int(sys.argv[3])
_displaySample = int(sys.argv[4])
_model = (sys.argv[5])
_class = (sys.argv[6])
_mp = int(sys.argv[7])

VAE = getattr(importlib.import_module(_model,package='VAE.VAE'),_class)
print(f'Job Detail : {_class} {_model} {_bsize} {_pool} {_round} {_mp} {_displaySample}')

_model = _model.split('.')[3]


root=f"../Model/R{_round}/"
if(not os.path.isdir(root)):
    os.mkdir(root)

imagedatasetpath='../AllArrow/'
tensorboardPath=f'../Tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+\
                f'_R{_round}_bsize{_bsize}_5e7d_{_model}_brightness2_solarize0_{_pool}_CoraseGrain'

writer = SummaryWriter(tensorboardPath)
epochs=500
trainingbatchsize=_bsize
validationbatchsize=_bsize

def reparametrize(mu,log_var):
    #Reparametrization Trick to allow gradients to backpropagate from the
    #stochastic part of the model
    sigma = torch.exp(0.5*log_var)
    z = torch.randn_like(log_var)
    #z= z.type_as(mu)
    return mu + sigma*z

def reparametrizeNP(mu,log_var):
    #Reparametrization Trick to allow gradients to backpropagate from the
    #stochastic part of the model
    sigma = np.exp(0.5*log_var)
    z = np.random.standard_normal(size=log_var.shape)
    #z= z.type_as(mu)
    return mu + sigma*z

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_output(name,storage):
    def hook(model, input, output):
        storage[name] = output.clone().detach().cpu().numpy()
    return hook

transformations1 = transforms.Compose([
    #transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0, hue=0),
    transforms.RandomRotation(359),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

transformations2 = transforms.Compose([
    #transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0, hue=0),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
imagepath=[]
for f in listdir(imagedatasetpath):
    if isfile(join(imagedatasetpath, f)):
        imagepath.append(imagedatasetpath + f)

imagepath=np.array(imagepath)
t=int(len(imagepath)*0.7)
idx = [x for x in range(len(imagepath))]
np.random.shuffle(idx)
trainidx = idx[:t]
validateidx = idx[t:]

trainingset = ImageDatasetPath(imagepath[trainidx],transformations1,transformations2,2,True,0)
validationset = ImageDatasetPath(imagepath[validateidx],transformations1,transformations2,2,True,0)

training_dataset_sizes = len(trainingset)
validation_dataset_sizes = len(validationset)
print(f"Training Size : {training_dataset_sizes}")
print(f"Validation Size : {validation_dataset_sizes}" )


trainloader = torch.utils.data.DataLoader(trainingset,batch_size=trainingbatchsize,shuffle=True)
validateloader = torch.utils.data.DataLoader(validationset,batch_size=validationbatchsize,shuffle=True)


model=VAE(pool=_pool)
summary(model,(3,256,256))

reconstructionlossfunction=torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

bestloss = float("Inf")

for epoch in range(epochs):
    trainingloss = 0
    trainingKLDloss = 0
    trainingreconstructionloss = 0
    trainingsloss=0
    validateloss = 0
    validationKLDloss=0
    validationreconstructionloss=0
    validationsloss=0
    start=time.time()
    model.train()
    i=0
    for i, data in enumerate(trainloader):
        #same = np.random.choice(2,1)
        s,t = data
        out,mu,v =model(s.to(model.device))
        rloss=reconstructionlossfunction(out,t.to(model.device))
        kldloss =  -0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
        totalloss=rloss+kldloss
        optimizer.zero_grad()
        totalloss.backward()
        optimizer.step()
        trainingloss += totalloss.item()
        trainingKLDloss += kldloss.item()
        trainingreconstructionloss += rloss.item()


    print(f"Training Epoch : {epoch:3d} | Total loss : {(trainingloss/i):.4f} | Reconstruction loss: {(trainingreconstructionloss/i):10.8f} | KLD loss: {(trainingKLDloss/i):10.8f}")

    corasegain={}
    handler=[]
    model.eval()
    base=0
    for name,layer in model.encoder.named_modules():
        if isinstance(layer, torch.nn.AvgPool2d):
            if(int(name)>base):
                base=int(name)
            handler.append(layer.register_forward_hook(get_output(f'encoder_Avg_{name}',corasegain)))
        if isinstance(layer, torch.nn.MaxPool2d):
            if(int(name)>base):
                base=int(name)
            handler.append(layer.register_forward_hook(get_output(f'encoder_Max_{name}',corasegain)))
    for name,layer in model.FCM.named_modules():
        if isinstance(layer, torch.nn.Linear):
            base+=1
            handler.append(layer.register_forward_hook(get_output(f'featureLinearM_{base}',corasegain)))
    for name,layer in model.FCV.named_modules():
        if isinstance(layer, torch.nn.Linear):
            base+=1
            handler.append(layer.register_forward_hook(get_output(f'featureLinearV_{base}',corasegain)))

    j=0
    out=None
    t=None
    v1=None
    sampleX=None
    sampleY=None
    numSample=_displaySample
    with torch.no_grad():
        for j, data in enumerate(validateloader):
            s,t=data
            out,mu,v =model(s.to(model.device))
            if(j==0):
                sampleX = s.clone().detach().cpu().numpy()
                sampleY = t.clone().detach().cpu().numpy()
                for h in handler:
                    h.remove()
            vrloss=reconstructionlossfunction(out,t.to(model.device))
            vkldloss = -0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
            vloss=vrloss+vkldloss
            validateloss+= vloss.data
            validationKLDloss += vkldloss.data
            validationreconstructionloss += vrloss.data
            v1 = reparametrize(mu,v)
            writer.add_histogram(f'F Histogram',v1, epoch*10+i)
            torch.cuda.empty_cache()
    #Gen Data
    modelendTime=time.time()
    for name,param in model.encoder.named_parameters():
        if("weight" in name):
            norm=torch.linalg.norm(param.grad,dim=1)
            S = torch.linalg.svdvals(param)
            S = S**2
            writer.add_histogram(f"Encoder {name} Weight EigenValues",S,epoch)
            writer.add_histogram(f"Encoder {name} Weigh Gradient Norm",norm,epoch)
    for name,param in model.decoder.named_parameters():
        if("weight" in name):
            norm=torch.linalg.norm(param.grad)
            writer.add_scalar(f"Decoder {name} Weigh Gradient Norm",norm,epoch)
    for name,param in model.FCM.named_parameters():
        if("weight" in name):
            norm=torch.linalg.norm(param.grad,dim=1)
            S = torch.linalg.svdvals(param)
            S = S**2
            writer.add_histogram(f"FCM {name} Weight EigenValues",S,epoch)
            writer.add_histogram(f"FCM {name} Weigh Gradient Norm",norm,epoch)
    for name,param in model.FCV.named_parameters():
        if("weight" in name):
            norm=torch.linalg.norm(param.grad,dim=1)
            S = torch.linalg.svdvals(param)
            S = S**2
            writer.add_histogram(f"FCV {name} Weight EigenValues",S,epoch)
            writer.add_histogram(f"FCV {name} Weigh Gradient Norm",norm,epoch)
    layerName=sorted(list(corasegain.keys()), key= lambda x:int(str(x).split("_")[-1]))
    layerMI=pymp.shared.dict()
    moments=pymp.shared.dict()
    Kevg=pymp.shared.dict()
    image=pymp.shared.dict()
    x_yMI = ee.mi(sampleX,sampleY)
    layerMI['X:Y']=x_yMI
    with pymp.Parallel(_mp) as p:
        for k in p.range(len(layerName)-1):
            perviousScale=None
            perviousName='X'
            MI=None
            if(k!=0):
                perviousScale=corasegain[layerName[k-1]]
                perviousName=layerName[k-1]
            else:
                perviousScale=sampleX
            name=layerName[k]
            sampleScale = corasegain[name][0:numSample]
            gray_scale = np.sum(sampleScale, 1)
            gray_scale = gray_scale / sampleScale.shape[1]
            if(k==len(layerName)-1):
                sampleScale = reparametrizeNP(sampleScale,corasegain[layerName[-1]])
            with p.lock:
                image[f'{name}']=[sampleScale,gray_scale]
            allScale=corasegain[name]
            tmp=kstat(allScale,4)
            with p.lock:
                moments[f'{name}_4thOrder']=tmp
            MI=ee.mi(perviousScale,allScale)
            MIname=f'{perviousName}:{layerName[k]}'
            yMI=ee.mi(sampleY,allScale)
            with p.lock:
                layerMI[MIname]=MI
                layerMI[f'I(Y:{name})']=yMI
            if(k==len(layerName)-1):
                MI=ee.mi(sampleX,allScale)
                with p.lock:
                    layerMI['I(X,C)']=MI
            if(k<len(layerName)-2):
                allScale=allScale.reshape(-1,allScale.shape[-2],allScale.shape[-1])
                S=np.array((allScale.shape[0],allScale.shape[1]))
                for e in range(allScale.shape[0]):
                    f = allScale[e]
                    C = np.cov(f,bias=True)
                    if(np.sum(np.isnan(C))==0 and np.sum(np.isinf(C))==0):
                        try:
                            _,L,_ = np.linalg.svd(C)
                            S[e]=L
                        except np.linalg.LinAlgError:
                            print('SVD dont converge')
                            pass
                    else:
                        print('C contains NAN or INF')
                with p.lock:
                    Kevg[f'{name}_covEigenValue']=S
            if(k==len(layerName)-2):
                S = np.zeros_like(allScale.shape[1])
                sampleScale = reparametrizeNP(allScale,corasegain[layerName[-1]])
                cov = np.cov(sampleScale,bias=True)
                if(np.sum(np.isnan(cov))==0 and np.sum(np.isinf(cov))==0):
                    try:
                        _,S,_ = np.linalg.svd(cov)
                    except np.linalg.LinAlgError:
                        print('SVD dont converge')
                        pass
                else:
                    print('SVD dont converge')
                with p.lock:
                    Kevg[f'{name}_covEigenValue']=S

    writer.add_scalars('MI',layerMI,epoch)
    writer.add_scalars('4thOrder',moments,epoch)
    for key in Kevg.keys():
        writer.add_histogram(key,np.array(Kevg[key]),epoch)

    for key in image.keys():
        sampleScale= torch.tensor(image[key][0])
        gray_scale = torch.tensor(image[key][1])
        img = make_grid(sampleScale,nrow=numSample,normalize=True,pad_value=0.8)
        img = img.unsqueeze(dim=1)
        if(len(gray_scale.shape)>1):
            gray_scale = make_grid(gray_scale,nrow=numSample,padding=5,normalize=True,pad_value=0.8)
            gray_scale = gray_scale.unsqueeze(dim=1)
            writer.add_image(f'{key} AvgFilter ({sampleScale.shape[-2]},{sampleScale.shape[-1]})', gray_scale, epoch,dataformats='NCHW')
        writer.add_image(f'{key}', img, epoch, dataformats='NCHW')

    end=time.time()
    writer.add_image("Validation (Decode)" , torch.logit(out) , epoch  , dataformats='NCHW')
    writer.add_image("Validation (Target)" , t , epoch  , dataformats='NCHW')
    #writer.add_image("Validation (Out)" , out , epoch  , dataformats='NCHW')
    writer.add_scalar('training loss',trainingloss/i,epoch)
    writer.add_scalar('training KLD loss',trainingKLDloss/i,epoch)
    writer.add_scalar('training reconstruction loss',trainingreconstructionloss/i,epoch)
    #writer.add_scalar('training S loss',trainingsloss/i,epoch)

    writer.add_scalar('validation loss',validateloss/j,epoch)
    writer.add_scalar('validation KLD loss',validationKLDloss/j,epoch)
    writer.add_scalar('validation reconstruction loss',validationreconstructionloss/j,epoch)
    #writer.add_scalar('validation S loss',validationsloss/j,epoch)


    print(f"Validation Epoch : {epoch:3d} | Total loss : {(validateloss/j):10.8f} | Reconstruction loss: {(validationreconstructionloss/j):10.8f} | KLD loss: {(validationKLDloss/j):10.8f} | Model Time :{modelendTime-start} | Data Processing Time : {end-modelendTime}")
    if(bestloss>(validateloss/j)):
        torch.save(model, root+"checkpoint_epoch_"+str(epoch)+"_"+str(trainingloss/i)+"_"+str(validateloss/j)+"_.pth")
        bestloss=(validateloss/j)
