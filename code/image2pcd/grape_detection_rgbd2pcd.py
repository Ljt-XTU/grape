# Author:ljt
# Time:2023/4/7 10:03
# Illustration:
import os
import numpy as np
import open3d as o3d
from detection_rgbd2pcd import getbboxes,hand_creat_pcd,cropimage
def getfilepath(filedir):
    filenames=os.listdir(filedir)
    filenames.sort()
    filepaths=[]
    for filename in filenames:
        filepaths.append(filedir+'/'+filename)
    return filepaths
def pcd_save(config,checkpoint,image_paths,color_paths,depth_paths):
    for image_path,depth_path,color_path in zip(image_paths,depth_paths,color_paths):
        depth_name=depth_path[depth_path.rfind('/')+1:]
        color_name=color_path[color_path.rfind('/')+1:]
        image_name=image_path[image_path.rfind('/')+1:]
        number=depth_name[depth_name.rfind('_')+1:depth_name.rfind('.')]
        save_dir=image_path[0:image_path.rfind('J')]+'GrapePCD/'
        print('\n-------------------------------------------------\n')
        print('depth name:{0}  color name:{1}  image name:{2}  number:{3}'.\
                        format(depth_name,color_name,image_name,number))
        bunch_bboxes = getbboxes(config_file, checkpoint_file, image_path)
        print('bunch bboxes:{0}'.format(bunch_bboxes))
        print('\n-------------------------------------------------\n')
        color_image = np.load(color_path)[:, :, ::-1]
        depth_image = np.load(depth_path)[-1]
        for idx,bbox in enumerate(bunch_bboxes):
            crop_color, crop_depth, leftup = cropimage(color_image, depth_image, bbox)
            pcd = hand_creat_pcd(crop_color, crop_depth, leftup)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.io.write_point_cloud('{0}grape_pcd_{1}({2}).pcd'.format(save_dir,number,idx),pcd)



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

