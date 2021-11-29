import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import Transformer,Discriminator
from utils import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init



from mocap_data import DATA
dataset = DATA()
batch_size=64

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

from discriminator_data import D_DATA
real_=D_DATA()

real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=batch_size, shuffle=True)
real_motion_all=list(enumerate(real_motion_dataloader))

device='cuda'



model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)

discriminator = Discriminator(d_word_vec=45, d_model=45, d_inner=256,
            n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)

lrate=0.0003
lrate2=0.0005

params = [
    {"params": model.parameters(), "lr": lrate}
]
optimizer = optim.Adam(params)
params_d = [
    {"params": discriminator.parameters(), "lr": lrate}
]
optimizer_d = optim.Adam(params_d)


for epoch in range(100):
    total_loss=0
    
    for j,data in enumerate(dataloader,0):
                
        use=None
        input_seq,output_seq=data
        input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # batch, N_person, 15 (15 fps 1 second), 45 (15joints xyz) 
        output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz) 
        
        # first 1 second predict future 1 second
        input_=input_seq.view(-1,15,input_seq.shape[-1]) # batch x n_person ,15: 15 fps, 1 second, 45: 15joints x 3
        
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

        input_ = dct.dct(input_)
                
        rec_=model.forward(input_[:,1:15,:]-input_[:,:14,:],dct.idct(input_[:,-1:,:]),input_seq,use)

        rec=dct.idct(rec_)

        # first 2 seconds predict 1 second
        new_input=torch.cat([input_[:,1:15,:]-input_[:,:14,:],dct.dct(rec_)],dim=-2)
        
        new_input_seq=torch.cat([input_seq,output_seq[:,:,1:16]],dim=-2)
        new_input_=dct.dct(new_input_seq.reshape(-1,30,45))
        new_rec_=model.forward(new_input_[:,1:,:]-new_input_[:,:29,:],dct.idct(new_input_[:,-1:,:]),new_input_seq,use)

        new_rec=dct.idct(new_rec_)

        # first 3 seconds predict 1 second
        new_new_input_seq=torch.cat([input_seq,output_seq[:,:,1:31]],dim=-2)
        new_new_input_=dct.dct(new_new_input_seq.reshape(-1,45,45))
        new_new_rec_=model.forward(new_new_input_[:,1:,:]-new_new_input_[:,:44,:],dct.idct(new_new_input_[:,-1:,:]),new_new_input_seq,use)

        new_new_rec=dct.idct(new_new_rec_)
        
        rec=torch.cat([rec,new_rec,new_new_rec],dim=-2)
        
        results=output_[:,:1,:]
        for i in range(1,31+15):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]
      
        loss=torch.mean((rec[:,:,:]-(output_[:,1:46,:]-output_[:,:45,:]))**2)
        
        
        if (j+1)%2==0:
            
            fake_motion=results

            disc_loss=disc_l2_loss(discriminator(fake_motion))
            loss=loss+0.0005*disc_loss
            
            fake_motion=fake_motion.detach()

            real_motion=real_motion_all[int(j/2)][1][1]
            real_motion=real_motion.view(-1,46,45)[:,1:46,:].float().to(device)

            fake_disc_value = discriminator(fake_motion)
            real_disc_value = discriminator(real_motion)

            d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value, fake_disc_value)
            
            optimizer_d.zero_grad()
            d_motion_disc_loss.backward()
            optimizer_d.step()
        
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss=total_loss+loss

    print('epoch:',epoch,'loss:',total_loss/(j+1))
    if (epoch+1)%5==0:
        save_path=f'./saved_model/{epoch}.model'
        torch.save(model.state_dict(),save_path)


        
        
