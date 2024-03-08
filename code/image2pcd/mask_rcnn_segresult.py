# Author:ljt
# Time:2023/7/12 11:00
# Illustration:
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import open3d as o3d
def getfilepath(filedir):
    filenames=os.listdir(filedir)
    filenames.sort()
    filepaths=[]
    for filename in filenames:
        filepaths.append(filedir+'/'+filename)
    return filepaths

def hand_creat_pcd(color_img,depth_img,crop_mask,crop_point):
    height,width=depth_img.shape
    crop_x,crop_y=crop_point   #top left point of crop area 截取区域左上角点
    #camera internal parameter 相机内参 
    fx,cx=913.1469116210938,640.0037841796875
    fy,cy=911.2598876953125,358.8372802734375
    depth_scale=1000    #depth scale深 度比例
    hand_pcd=o3d.geometry.PointCloud()

    #generate image's coordinate 生成图像(u,v)坐标
    vec_u=np.linspace(crop_x,crop_x+height-1,height)
    vec_v=np.linspace(crop_y,crop_y+width-1,width)
    V,U=np.meshgrid(vec_v,vec_u)

    #caculate 计算（u,v）-> (x,y)
    Z=depth_img/depth_scale
    X,Y=(V-cx)*Z/fx,(U-cy)*Z/fy

    P=np.dstack((X,Y,Z)).reshape(-1,3)  #point cloud coordinate 点云坐标点
    C=(color_img/255).reshape(-1,3) #point cloude color 点云颜色
    mask_label=crop_mask.reshape(-1,1)
    #select the points that distance from camera less than 3m 选取距离小于3m的点
    indx=np.where((P[:,2]<=3.0))
    points,colors=P[indx],C[indx]
    mask_label=mask_label[indx]
    point=o3d.utility.Vector3dVector(np.asarray(points))
    color=o3d.utility.Vector3dVector(np.asarray(colors))
    hand_pcd.points,hand_pcd.colors=point,color
    return hand_pcd,mask_label

def cropimage(color,depth,sigle_mask,bbox):
    crop_color=color[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    crop_depth=depth[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    crop_mask=sigle_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    leftup_point=bbox[:2]
    return crop_color,crop_depth,crop_mask,leftup_point
#get grape bbx 获取目标检测中的每串葡萄框
def getbboxes(config_file,checkpoint_file,image_file):
    device='cuda:0'
    score_thr=0.35
    #get model 获取模型
    model = init_detector(config_file, checkpoint_file, device=device)
    result = inference_detector(model, image_file)    #detect grapes and peduncles 对输入图片检测葡萄和果梗

    #get labels and bbxes 获取标签和框
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    grape_segm,pedun_segm=np.asarray(segm_result[0]),np.asarray(segm_result[1])
    all_segm=np.concatenate((grape_segm,pedun_segm),axis=0)
    # bbox format[left_top_x,left_top_y,right_down_x,right_down_y,confidence_level(score)]
    bboxes ,labels= np.vstack(bbox_result),np.concatenate(labels)
    print('bboxes:{0}\nlabels:{1}'.format(bboxes, labels))
    print('grape_segm shape:{0}\npedun segm shape:{1}'.format(grape_segm.shape,pedun_segm.shape))
    print('all_segm shape:{0}'.format(all_segm.shape))
    #remove the low confidence bbxes by thredshold 按阈值去除置信度低的框
    trust_index=bboxes[:,-1]>score_thr
    tru_bboxes,tru_labels=bboxes[trust_index],labels[trust_index]
    tru_segm=all_segm[trust_index]
    print('tru_bboxes:{0}\ntru_labels:{1}\ntru_segm shape:{2}'.\
          format(tru_bboxes,tru_labels,tru_segm.shape))
    #get the bbxes of grapes and peduncles by labels 按标签获取葡萄和果梗的框
    grape_bboxes,peduncle_bboxes=tru_bboxes[tru_labels==0],tru_bboxes[tru_labels==1]
    grape_mask,pedun_mask=tru_segm[tru_labels==0],tru_segm[tru_labels==1]
    print('grape_bboxes:{0}\npeduncle_labels:{1}'.format(grape_bboxes,peduncle_bboxes))
    print('grape_mask shape:{0}\npedun_mask shape:{1}'.format(grape_mask.shape,pedun_mask.shape))
    #get the bif bbx of a bunch of grape 获取一串葡萄(果串+果梗)的大框
    bunch_box=[]
    mask=[]
    for indx_grape,gra_box in enumerate(grape_bboxes):
        flag=0
        for indx_pedun,ped_box in enumerate(peduncle_bboxes):
            if (ped_box[0]>gra_box[0] and ped_box[2]<gra_box[2]):
                bunch_box.append([gra_box[0],ped_box[1],gra_box[2],gra_box[3]])
                grape_label=grape_mask[indx_grape].astype(np.int32)
                pedun_label=np.where(pedun_mask[indx_pedun].astype(np.int32)==1,2,0)
                mask.append(grape_label+pedun_label)
                flag=1
        if (flag==0):
            bunch_box.append(gra_box[:-1].tolist())
            mask.append(grape_mask[indx_grape].astype(np.int32))

    #show_result_pyplot(model, img_path, result)
    return np.asarray(bunch_box,dtype=np.int32),np.asarray(mask)

def pcd_save(config,checkpoint,image_paths,color_paths,depth_paths):
    for image_path,depth_path,color_path in zip(image_paths,depth_paths,color_paths):
        depth_name=depth_path[depth_path.rfind('/')+1:]
        color_name=color_path[color_path.rfind('/')+1:]
        image_name=image_path[image_path.rfind('/')+1:]
        number=depth_name[depth_name.rfind('_')+1:depth_name.rfind('.')]
        save_dir=image_path[0:image_path.rfind('J')]+'MaskLabel/'
        print('\n-------------------------------------------------\n')
        print('depth name:{0}  color name:{1}  image name:{2}  number:{3}'.\
                        format(depth_name,color_name,image_name,number))
        bunch_bboxes,mask = getbboxes(config, checkpoint, image_path)
        print('bunch bboxes:{0}'.format(bunch_bboxes))
        print('mask shape:{0}'.format(mask.shape))
        print('\n-------------------------------------------------\n')
        color_image = np.load(color_path)[:, :, ::-1]
        depth_image = np.load(depth_path)[-1]
        for idx,(bbox,sigle_mask) in enumerate(zip(bunch_bboxes,mask)):
            crop_color, crop_depth,crop_mask, leftup = cropimage(color_image, depth_image,sigle_mask, bbox)
            print('crop_mask shape:{0}'.format(crop_mask.shape))
            pcd ,mask_label= hand_creat_pcd(crop_color, crop_depth,crop_mask, leftup)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            np.savetxt('{0}grape_masklabel_{1}({2}).txt'.format(save_dir,number,idx),mask_label,fmt='%.3f')
            #o3d.io.write_point_cloud('{0}grape_pcd_{1}({2}).pcd'.format(save_dir,number,idx),pcd)



if __name__=='__main__':
    config_file = '../work_dirs/mask_rcnn_r50_fpn_1x_grape/mask_rcnn_r50_fpn_1x_grape.py'
    checkpoint_file = '../work_dirs/mask_rcnn_r50_fpn_1x_grape/latest.pth'
    train_color_dir='../train/ColorNpy'
    train_depth_dir='../train/DepthNpy'
    train_image_dir='../train/JPEGImages'
    val_color_dir='../val/ColorNpy'
    val_depth_dir='../val/DepthNpy'
    val_image_dir='../val/JPEGImages'
    color_paths=getfilepath(val_color_dir)
    depth_paths=getfilepath(val_depth_dir)
    image_paths=getfilepath(val_image_dir)
    pcd_save(config_file,checkpoint_file,image_paths,color_paths,depth_paths)

