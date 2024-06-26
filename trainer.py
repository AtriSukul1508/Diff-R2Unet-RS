import torch
from torch.optim import SGD
import os
from data import HSI_Loader
from r2unet import R2UNet
from diffusion import Diffusion,device
import os


EPOCH = 3000 
lr = 1e-3
T=1000

datasetName = 'Pavia'
batch_size = 64
patch_size = 16
selectBands = 'all'
NoOfBands = 104


path = f"./model/{datasetName}_diff" 

def train():
    if not os.path.exists(path):
        os.makedirs(path)
        
        
    dataloader = HSI_Loader({"data":{"dataset":datasetName}})
    trainLoader,*rest = dataloader.dataset()
    diffusion = Diffusion(T)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    
    model = R2UNet(_image_channels=1)
    model.to(device)


    for epoch in range(EPOCH):
        for step, (batch, _) in enumerate(trainLoader):
            optimizer.zero_grad()
            
            batch = batch.to(device)
            t = torch.randint(0, diffusion.T , (batch.shape[0],), device=device).long()
            loss = diffusion.loss_func(model, batch, t)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch: {epoch} -- step: {step:03d} Loss: {loss.item()} ")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"{path}/PU_R2Unet_{epoch}.pkl")
            print("Model saved")


if __name__ == "__main__":
    train()
