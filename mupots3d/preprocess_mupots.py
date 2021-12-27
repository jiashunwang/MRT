import open3d as o3d 
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import trimesh

################
# 3 persons
final_data=[]
for j in range(1,21,1):


    data=io.loadmat('./data/TS'+str(j)+'/annot.mat')['annotations']
    if data.shape[1]!=3:
        continue


    #print(j)
    
    print(data.shape)
    
    v_total=[]
    for i in range(len(data)):
        v1=data[i][0][0][0][1].transpose(1,0)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3,:3]
        v1=np.matmul(v1,rot)
        v2=data[i][1][0][0][1].transpose(1,0)
        #rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])[:3,:3]
        v2=np.matmul(v2,rot)
        v3=data[i][2][0][0][1].transpose(1,0)
        #rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])[:3,:3]
        v3=np.matmul(v3,rot)
        v=np.concatenate([v1,v2,v3]).reshape(3,17,3)
        v_total.append(v)
    temp=np.array(v_total)
    temp=temp.swapaxes(0,1)
    temp[:,:,:,1]=temp[:,:,:,1]-np.min(temp[:,:,:,1]) #foot on ground
    temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0]) #center
    temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) #center
    use=[14,11,12,13,8,9,10,1,0,5,6,7,2,3,4] #used joints and order
    temp_data=temp[:,:,use,:] 
    
    for i in range(0,temp_data.shape[1],15): #down sample

        if (i+120)>temp_data.shape[1]:
            break
        final_data.append(temp_data[:,i:i+120,:,:])
    #print(j)
final_data=np.concatenate(final_data)*0.017*0.1*1.8/3 # scale
final_data=final_data.reshape(-1,3,120,45) # n, 3 persons, 30 fps 4 seconds, 15 joints  xyz coordinates

np.save('mupots_120_3persons.npy',final_data)

################
# 2 persons

final_data=[]
for j in range(1,21,1):

    
    data=io.loadmat('./data/TS'+str(j)+'/annot.mat')['annotations']
    
    if data.shape[1]==3:
        continue

    #print(j)

    print(data.shape)

    v_total=[]
    for i in range(len(data)):
        v1=data[i][0][0][0][1].transpose(1,0)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3,:3]
        v1=np.matmul(v1,rot)
        v2=data[i][1][0][0][1].transpose(1,0)
        #rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])[:3,:3]
        v2=np.matmul(v2,rot)
        
        v=np.concatenate([v1,v2]).reshape(2,17,3)
        v_total.append(v)
    temp=np.array(v_total)
    temp=temp.swapaxes(0,1)
    temp[:,:,:,1]=temp[:,:,:,1]-np.min(temp[:,:,:,1]) #foot on ground
    temp[:,:,:,0]=temp[:,:,:,0]-np.mean(temp[:,:,:,0]) #center
    temp[:,:,:,2]=temp[:,:,:,2]-np.mean(temp[:,:,:,2]) #center
    use=[14,11,12,13,8,9,10,1,0,5,6,7,2,3,4] #used joints and order
    temp_data=temp[:,:,use,:] 
    
    for i in range(0,temp_data.shape[1],15): #down sample

        if (i+120)>temp_data.shape[1]:
            break
        final_data.append(temp_data[:,i:i+120,:,:])
    #print(j)
final_data=np.concatenate(final_data)*0.017*0.1*1.8/3 # scale
final_data=final_data.reshape(-1,2,120,45) # n, 2 persons, 30 fps 4 seconds, 15 joints xyz coordinates

np.save('mupots_120_2persons.npy',final_data)