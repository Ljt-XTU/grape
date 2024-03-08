# Author:ljt
# Time:2023/4/5 10:36
# Illustration:
import open3d as o3d
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import numpy as np
import cv2 as cv

#获取目标检测中的每串葡萄框
def getbboxes(config_file,checkpoint_file,image_file):
    device='cuda:0'
    score_thr=0.35
    #获取模型
    model = init_detector(config_file, checkpoint_file, device=device)
    result = inference_detector(model, image_file)    #对输入图片检测葡萄和果梗

    #获取标签和框
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    # bbox format[left_top_x,left_top_y,right_down_x,right_down_y,confidence_level(score)]
    bboxes ,labels= np.vstack(bbox_result),np.concatenate(labels)
    print('bboxes:{0}\nlabels:{1}'.format(bboxes, labels))

    #按阈值去除置信度低的框
    trust_index=bboxes[:,-1]>score_thr
    tru_bboxes,tru_labels=bboxes[trust_index],labels[trust_index]
    print('tru_bboxes:{0}\ntru_labels:{1}'.format(tru_bboxes,tru_labels))

    #按标签获取葡萄和果梗的框
    grape_bboxes,peduncle_bboxes=tru_bboxes[tru_labels==0],tru_bboxes[tru_labels==1]
    print('grape_bboxes:{0}\npeduncle_labels:{1}'.format(grape_bboxes,peduncle_bboxes))

    #获取一串葡萄(果串+果梗)的大框
    bunch_box=[]
    for gra_box in grape_bboxes:
        flag=0
        for ped_box in peduncle_bboxes:
            if (ped_box[0]>gra_box[0] and ped_box[2]<gra_box[2]):
                bunch_box.append([gra_box[0],ped_box[1],gra_box[2],gra_box[3]])
                flag=1
        if (flag==0):
            bunch_box.append(gra_box[:-1].tolist())

    #show_result_pyplot(model, img_path, result)

    return np.asarray(bunch_box,dtype=np.int32)

def cropimage(color,depth,bbox):
    crop_color=color[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    crop_depth=depth[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    leftup_point=bbox[:2]
    return crop_color,crop_depth,leftup_point

def hand_creat_pcd(color_img,depth_img,crop_point):
    height,width=depth_img.shape
    crop_x,crop_y=crop_point   #截取区域左上角点
    #相机内参
    fx,cx=913.1469116210938,640.0037841796875
    fy,cy=911.2598876953125,358.8372802734375
    depth_scale=1000    #深度比例
    hand_pcd=o3d.geometry.PointCloud()

    #生成图像(u,v)坐标
    vec_u=np.linspace(crop_x,crop_x+height-1,height)
    vec_v=np.linspace(crop_y,crop_y+width-1,width)
    V,U=np.meshgrid(vec_v,vec_u)

    #计算（u,v）-> (x,y)
    Z=depth_img/depth_scale
    X,Y=(V-cx)*Z/fx,(U-cy)*Z/fy

    P=np.dstack((X,Y,Z)).reshape(-1,3)  #点云坐标点
    C=(color_img/255).reshape(-1,3) #点云颜色

    #选取距离小于3m的点
    indx=np.where((P[:,2]<=3.0))
    points,colors=P[indx],C[indx]
    point=o3d.utility.Vector3dVector(np.asarray(points))
    color=o3d.utility.Vector3dVector(np.asarray(colors))
    hand_pcd.points,hand_pcd.colors=point,color
    return hand_pcd

if __name__=='__main__':
    config_file = '../work_dirs/mask_rcnn_r50_fpn_1x_grape/mask_rcnn_r50_fpn_1x_grape.py'
    checkpoint_file = '../work_dirs/mask_rcnn_r50_fpn_1x_grape/latest.pth'
    img_path='../train/JPEGImages/grape_image_104.jpg'
    color_path='../train/ColorNpy/grape_color_104.npy'
    depth_path='../train/DepthNpy/grape_depths_104.npy'
    bunch_bboxes=getbboxes(config_file,checkpoint_file,img_path)
    print(bunch_bboxes)
    color_image=np.load(color_path)[:,:,::-1]
    depth_image=np.load(depth_path)[-1]
    for bbox in bunch_bboxes:
        crop_color,crop_depth,leftup=cropimage(color_image,depth_image,bbox)
        cv.imshow('image',crop_color)
        cv.waitKey()
        pcd=hand_creat_pcd(crop_color,crop_depth,leftup)
    o3d.visualization.draw(pcd)


    print('color_image:{0}\ndepths_image:{1}'.format(color_image.shape,depth_image.shape))