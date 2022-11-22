
import os
import os.path as osp
import shutil
import numpy as np
from glob import glob

P_SEG = "/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/bbox"
DEST="/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected"

def copy_first():
    DEST="/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected"

    os.makedirs(DEST, exist_ok=True)

    dir_list = os.listdir(P_SEG)
    

    for _dir in dir_list:
        seg_path = osp.join(P_SEG, _dir, '00000.png')
        shutil.copy(seg_path, osp.join(DEST, _dir))

        bbox_path = osp.join(P_SEG, _dir, '00000.npy')
        shutil.copyfile(bbox_path, osp.join(DEST, _dir.split(".")[0]+'.npy'))

        

def copy_second():
    DEST="/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected"

    os.makedirs(DEST, exist_ok=True)

    dir_list = [42, 96, 98, 100, 102, 103, 114, 116, 117, 119, 120, 124, 126, 127, 131, 132, 134]

    dir_list = [str(id).zfill(5) + '.png' for id in dir_list]
    

    for _dir in dir_list:
        seg_path = osp.join(P_SEG, _dir, '00001.png')
        shutil.copy(seg_path, osp.join(DEST, _dir))

        bbox_path = osp.join(P_SEG, _dir, '00001.npy')
        shutil.copyfile(bbox_path, osp.join(DEST, _dir.split(".")[0]+'.npy'))

def copy_third():
    DEST="/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected"

    os.makedirs(DEST, exist_ok=True)

    dir_list = [117, 120]

    dir_list = [str(id).zfill(5) + '.png' for id in dir_list]
    

    for _dir in dir_list:
        seg_path = osp.join(P_SEG, _dir, '00002.png')
        shutil.copy(seg_path, osp.join(DEST, _dir))

        bbox_path = osp.join(P_SEG, _dir, '00002.npy')
        shutil.copyfile(bbox_path, osp.join(DEST, _dir.split(".")[0]+'.npy'))

def make_empty():
    DEST="/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected"

    os.makedirs(DEST, exist_ok=True)

    dir_list = [107, 108, 109, 110, 111, 112, 113]

    dir_list = [str(id).zfill(5) + '.png' for id in dir_list]
    
    seg_path = osp.join(P_SEG, '00001.png', '00000.png')
    import cv2
    img = cv2.imread(seg_path)
    save_img = img * 0

    for _dir in dir_list:
        cv2.imwrite(osp.join(DEST, _dir), save_img)
        os.remove(osp.join(DEST, _dir.split(".")[0]+'.npy'))



def make_gif():
    import imageio
    import cv2

    png_list = glob(osp.join(DEST, "*.png"))
    png_list.sort()
    imgs = [ cv2.imread(i) for i in png_list]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imageio.mimsave('/home/inhee/VCL/insect_recon/vision_tools/debug/test.gif', imgs, fps=10.0)



def make_gif():
    import imageio
    import cv2

    png_list = glob('/home/inhee/VCL/insect_recon/ngp-pifu/_pifu_ngp/selected_mask/results/*_rgb.png')
    png_list.sort()
    imgs = [ cv2.imread(i) for i in png_list]
    imageio.mimsave('/home/inhee/VCL/insect_recon/vision_tools/debug/test2.gif', imgs, fps=10.0)
    

if __name__ == '__main__':
    #copy_first()
    #copy_second()
    #copy_third()
    #make_empty()
    make_gif()

