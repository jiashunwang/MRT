import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=np.load('two_train_4seconds_2.npy',allow_pickle=True)
eg=1
data_list=data[eg]

data_list=data_list.reshape(-1,120,31,3)
data_list=data_list*0.1*1.8/3 # scale
#no need to scale if using the mix_mocap data

body_edges = np.array(
[[0,1], [1,2],[2,3],[3,4],
[4,5],[0,6],[6,7],[7,8],[8,9],[9,10],[0,11],[11,12],[12,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[21,22],[20,23],[13,24],[24,25],[25,26],[26,27],[27,28],[28,29],[27,30]]
)

'''
if use the 15 joints in common
use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
data_list=data_list.reshape(-1,120,15,3)
data_list=data_list[:,:,[0,1,4,7,2,5,8,12,15,16,18,20,17,19,21],:]
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)
'''

fig = plt.figure(figsize=(10, 4.5))
            
ax = fig.add_subplot(111, projection='3d')

plt.ion()


length_=data_list.shape[1]

i=0
while i < length_:
    ax.lines = []
    for j in range(len(data_list)):
        
        xs=data_list[j,i,:,0]
        ys=data_list[j,i,:,1]
        zs=data_list[j,i,:,2]
        #print(xs)
        ax.plot( zs,xs, ys, 'y.')
        
        
        plot_edge=True
        if plot_edge:
            for edge in body_edges:
                x=[data_list[j,i,edge[0],0],data_list[j,i,edge[1],0]]
                y=[data_list[j,i,edge[0],1],data_list[j,i,edge[1],1]]
                z=[data_list[j,i,edge[0],2],data_list[j,i,edge[1],2]]
                if i>=30:
                    ax.plot(z,x, y, 'green')
                else:
                    ax.plot(z,x, y, 'blue')
        
        
        ax.set_xlim3d([-2 , 2])
        ax.set_ylim3d([-2 , 2])
        ax.set_zlim3d([-0, 2])
        #ax.set_axis_off()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    plt.pause(0.01)
    i += 1
    
    
plt.ioff()
plt.show()