import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=np.load('mupots_120_3persons.npy',allow_pickle=True)
eg=1
data_list=data[eg]

data_list=data_list.reshape(-1,120,15,3)




body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)


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