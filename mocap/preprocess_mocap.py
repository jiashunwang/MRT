from amc_parser import *
import numpy as np
import os

#two subjects data

data=[]
test_data=[]
for ii in range(4):

    # 18 19 20 21 22 23 33 34 are two subjects data
    if ii==0:
        A='18'
        B='19'
    if ii==1:
        A='20'
        B='21'
    if ii==2:
        A='22'
        B='23'
    if ii==3:
        A='33'
        B='34'

    motion_list_A_All=[]
    motion_list_A_test=[]
    asf_path = './all_asfamc/subjects/'+A+'/'+A+'.asf'
    iii=0
    for each in os.listdir('./all_asfamc/subjects/'+A+'/'):
        if each[-3:]!='amc':
            continue
        print(each)
        amc_path = './all_asfamc/subjects/'+A+'/'+each
        joints = parse_asf(asf_path)
        motions = parse_amc(amc_path)
        length=len(motions)
        
        if (iii%4==1) and (ii!=3): #just an example
            print('test')
            motion_list_A=[]
            for i in range(0,length,4):
                frame_idx = i
                joints['root'].set_motion(motions[frame_idx])
                joints_list=[]
                for joint in joints.values():
                    xyz=np.array([joint.coordinate[0],\
                        joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
                    joints_list.append(xyz)
                motion_list_A.append(np.array(joints_list))
            motion_list_A_test.append(motion_list_A)
        

        else:
            if ii==3 and iii%4==1:
                continue
            
            print('train')
            motion_list_A=[]
            for i in range(0,length,4):
                frame_idx = i
                joints['root'].set_motion(motions[frame_idx])
                joints_list=[]
                for joint in joints.values():
                    xyz=np.array([joint.coordinate[0],\
                        joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
                    joints_list.append(xyz)
                motion_list_A.append(np.array(joints_list))
            motion_list_A_All.append(motion_list_A)

        iii=iii+1

    motion_list_B_All=[]
    motion_list_B_test=[]
    asf_path_2 = './all_asfamc/subjects/'+B+'/'+B+'.asf'
    iii=0
    for each in os.listdir('./all_asfamc/subjects/'+B+'/'):
        if each[-3:]!='amc':
            continue
        print(each)
        amc_path_2 = './all_asfamc/subjects/'+B+'/'+each
        joints_2 = parse_asf(asf_path_2)
        motions_2 = parse_amc(amc_path_2)
        length=len(motions_2)
        
        if (iii%4==1) and (ii!=3):
            print('test')
            motion_list_B=[]
            for i in range(0,length,4):
                frame_idx = i
                joints_2['root'].set_motion(motions_2[frame_idx])
                joints_list_2=[]
                for joint in joints_2.values():
                    xyz=np.array([joint.coordinate[0],\
                        joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
                    joints_list_2.append(xyz)
                motion_list_B.append(np.array(joints_list_2))
            motion_list_B_test.append(motion_list_B)

        else:
            if ii==3 and iii%4==1:
                continue
            
            print('train')
            motion_list_B=[]
            for i in range(0,length,4):
                frame_idx = i
                joints_2['root'].set_motion(motions_2[frame_idx])
                joints_list_2=[]
                for joint in joints_2.values():
                    xyz=np.array([joint.coordinate[0],\
                        joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
                    joints_list_2.append(xyz)
                motion_list_B.append(np.array(joints_list_2))
            motion_list_B_All.append(motion_list_B)
        
        iii=iii+1

    scene_length=len(motion_list_B_All)

    #print(scene_length)
    for i in range(scene_length):
        motion_list_A=np.array(motion_list_A_All[i])
        motion_list_B=np.array(motion_list_B_All[i])
        #print(motion_list_A.shape[0])
        for j in range(0,motion_list_A.shape[0],2):
            
            if j+120>motion_list_A.shape[0]:
                break
            A_=np.expand_dims(np.array(motion_list_A[j:j+120]),0)
            B_=np.expand_dims(np.array(motion_list_B[j:j+120]),0)
            motion=np.concatenate([A_,B_])
            data.append(motion)

    scene_length=len(motion_list_B_test)
    for i in range(scene_length):
        motion_list_A=np.array(motion_list_A_test[i])
        motion_list_B=np.array(motion_list_B_test[i])
        #print(motion_list_A.shape[0])
        for j in range(0,motion_list_A.shape[0],2): #down sample
            
            if j+120>motion_list_A.shape[0]:
                break
            A_=np.expand_dims(np.array(motion_list_A[j:j+120]),0) # 120: 30 fps, 4 seconds
            B_=np.expand_dims(np.array(motion_list_B[j:j+120]),0)
            motion=np.concatenate([A_,B_])
            test_data.append(motion)
    print(ii)

np.save('two_train_4seconds_2.npy',np.array(data))
np.save('two_test_4seconds_2.npy',np.array(test_data))

########################################################################

#one subject data

data=[]
test_data=[]
for ii in os.listdir('./all_asfamc/subjects/'):
    
    motion_list_A_All=[]
    motion_list_A_test=[]
    asf_path = './all_asfamc/subjects/'+ii+'/'+ii+'.asf'
    iii=0
    for each in os.listdir('./all_asfamc/subjects/'+ii+'/'):
        if each[-3:]!='amc':
            continue
        amc_path = './all_asfamc/subjects/'+ii+'/'+each
        joints = parse_asf(asf_path)
        motions = parse_amc(amc_path)
        length=len(motions)
        if iii%4!=1:
            print('train')
            motion_list_A=[]
            for i in range(0,length,4):
                frame_idx = i
                joints['root'].set_motion(motions[frame_idx])
                joints_list=[]
                for joint in joints.values():
                    xyz=np.array([joint.coordinate[0],\
                        joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
                    joints_list.append(xyz)
                motion_list_A.append(np.array(joints_list))
            motion_list_A_All.append(motion_list_A)
        else:
            print('test')
            motion_list_A=[]
            for i in range(0,length,4):
                frame_idx = i
                joints['root'].set_motion(motions[frame_idx])
                joints_list=[]
                for joint in joints.values():
                    xyz=np.array([joint.coordinate[0],\
                        joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
                    joints_list.append(xyz)
                motion_list_A.append(np.array(joints_list))
            motion_list_A_test.append(motion_list_A)
        iii=iii+1
    scene_length=len(motion_list_A_All)
    for i in range(scene_length):
        motion_list_A=np.array(motion_list_A_All[i]) 
        for j in range(0,motion_list_A.shape[0],30): #down sample
            if (j+120)>motion_list_A.shape[0]:
                break
            A=np.expand_dims(np.array(motion_list_A[j:j+120]),0)            
            data.append(A)
    
    scene_length=len(motion_list_A_test)
    for i in range(scene_length):
        motion_list_A=np.array(motion_list_A_test[i])
        for j in range(0,motion_list_A.shape[0],30):
            if (j+120)>motion_list_A.shape[0]:
                break
            A=np.expand_dims(np.array(motion_list_A[j:j+120]),0)            
            test_data.append(A)
    print(ii)
np.save('one_train_4seconds_30.npy',np.array(data))
np.save('one_test_4seconds_30.npy',np.array(test_data))


#use mix_mocap.py to mix two subjects and one subject
