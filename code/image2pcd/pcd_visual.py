# Author:ljt
# Time:2023/4/10 9:10
# Illustration:
import open3d as o3d
if __name__=='__main__':
    pcd_path='../train/GrapePCD/grape_pcd_39(1).pcd'
    pcd=o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw(pcd)