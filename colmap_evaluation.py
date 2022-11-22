#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

# TODO : add intermediate starting methods
# TODO : develop it as general video preprocessing system.

'''
Apply colmap on directory.
'''

import argparse
import sys
import os
import os.path as osp
import glob
import shutil
import cv2
import math
import subprocess
import numpy as np
import ray
from tqdm import tqdm


'''some fixed settings'''
MIN_IMG = 10            # minimum required image, for single colmap reconstruction
TRIM_FPS = 3            # video trimming FPS
SKIP_FRAMES = 0         # skip first N frames when trimming

ENFORCE_SINGLE_MODEL = True                             # colmap mapper, enforcing single model
BMD_DIR = "/neuman/preprocess/BoostingMonocularDepth"   # Boost Monocular Depth estimation (fixed directory)



def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)


# from colmap2nerf file.
def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])




def get_points(txt_file):
    xyzs = np.empty((1,4))
    rgbs = np.empty((1,3), dtype=np.uint8)

    i = 0
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            
            elems=line.split(" ")

            xyz = np.array(tuple(map(float, elems[1:4]+[1]))).reshape(1,4)
            rgb = np.array(tuple(map(int, elems[4:7]))).reshape(1,3)
            rgb = rgb[:,::-1]

            xyzs = np.concatenate([xyzs, xyz], axis = 0)
            rgbs = np.concatenate([rgbs, rgb], axis = 0)
        
    
    xyzs = xyzs[1:,:].T
    rgbs = rgbs[1:,:]

    return xyzs, rgbs


def get_camera(txt_file):
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue

            elems = line.split(" ")

            if elems[1] != "PINHOLE":
                print("after undistortion, it should be PINHOLE!")
                assert()

            width = float(elems[2])
            height = float(elems[3])


            # in case of PINHOLE, it has following four variables
            fl_x = float(elems[4])
            fl_y = float(elems[5])

            cx = float(elems[6])
            cy = float(elems[7])

            # build camera projection matrix.
            # here, I simply built the camera projection matrix as 3x4

            intrinsic = [
                [fl_x, 0, cx, 0],
                [0, fl_y, cy, 0],
                [0, 0, 1, 0],
            ]

            return np.array(intrinsic), width, height




def projector(txt_dir, image_dir, res_dir):
    '''
    We will use undistorted image to evaluate.
    In other world. no need to take care of camera.
    '''
    os.makedirs(res_dir, exist_ok = True)

    xyzs, rgbs = get_points(osp.join(txt_dir, "points3D.txt"))
    intrinsic, w, h = get_camera(osp.join(txt_dir,"cameras.txt"))

    with open(os.path.join(txt_dir,"images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        up = np.zeros(3)
        
        
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
				#name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
				# why is this requireing a relitive path while using ^
                image_path = osp.join(image_dir, '_'.join(elems[9:]))

                # get_rotation
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                
                c2w = m
                #c2w = np.linalg.inv(c2w)
                # now it can applied to change the points

                modified_xyzs = np.matmul(c2w, xyzs)
                modified_xyzs = modified_xyzs / modified_xyzs[3,:]

                xy = np.matmul(intrinsic, modified_xyzs).T
                zs = xy[:,2:]
                xy = xy[:,0:2] / zs
                xy = np.round(xy)
                xy = xy.astype(np.int32)

                in_img = (xy[:,0]>=0) * (xy[:,0]<w) * (xy[:,1]>=0) * (xy[:,1]<h) * (zs[:,0]>0)

                print("{}, has {} points to be projected".format('_'.join(elems[9:]), in_img.sum()))

                # plot the points
                # for simplicity, Here I rounded whole points.

                proj = np.zeros((int(w),int(h),4))

                for cnt, _in in tqdm(enumerate(in_img)):
                    if _in == 0:
                        continue

                    proj[xy[cnt,0],xy[cnt,1],0:3] += rgbs[cnt]


                    for j in range(49):
                        dx = j%5 - 3
                        dy = j//5 - 3

                        if xy[cnt,0]+dx >= proj.shape[0] or xy[cnt,1]+dy >= proj.shape[1]:
                            continue
                        if xy[cnt,0]+dx < 0 or xy[cnt,1]+dy < 0:
                            continue
                        

                        proj[xy[cnt,0]+dx,xy[cnt,1]+dy,0:3] += rgbs[cnt]
                        proj[xy[cnt,0]+dx,xy[cnt,1]+dy,3] += 1

                proj = proj + 1e-8
                proj = proj / proj[:,:,3:4]

                proj = proj[:,:,0:3]

                proj = np.transpose(proj, [1, 0, 2])

                # read images
                org_img = cv2.imread(image_path)

                # overlay images
                overlay = org_img * 0.5 + proj * 0.5

                # merge images
                tot_img = np.concatenate([org_img, proj, overlay], axis = 1)

                # save results
                res_path = osp.join(res_dir, '_'.join(elems[9:]))

                cv2.imwrite(res_path, tot_img)




def ray_projector(txt_dir, image_dir, res_dir):
    '''
    We will use undistorted image to evaluate.
    In other world. no need to take care of camera.
    '''
    os.makedirs(res_dir, exist_ok = True)
    num_process=16
    ray.init(num_cpus=num_process, num_gpus=1)


    xyzs, rgbs = get_points(osp.join(txt_dir, "points3D.txt"))
    intrinsic, w, h = get_camera(osp.join(txt_dir,"cameras.txt"))

    with open(os.path.join(txt_dir,"images.txt"), "r") as f:
        i = 0
        up = np.zeros(3)
        
        ray_lists = []
        cpu_cnt = 0
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if  i % 2 == 1:
                ray_ret = ray_project_to_img.remote(line, image_dir, res_dir, intrinsic, xyzs, w, h, rgbs)
                ray_lists.append(ray_ret)
                cpu_cnt+=1
            
        ret = ray.get(ray_lists)



@ray.remote
def ray_project_to_img(line, image_dir, res_dir, intrinsic, xyzs, w, h, rgbs):
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

    elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
    #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
    # why is this requireing a relitive path while using ^
    image_path = osp.join(image_dir, '_'.join(elems[9:]))

    # get_rotation
    qvec = np.array(tuple(map(float, elems[1:5])))
    tvec = np.array(tuple(map(float, elems[5:8])))
    R = qvec2rotmat(qvec)
    t = tvec.reshape([3,1])
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    
    c2w = m
    #c2w = np.linalg.inv(c2w)
    # now it can applied to change the points

    modified_xyzs = np.matmul(c2w, xyzs)
    modified_xyzs = modified_xyzs / modified_xyzs[3,:]

    xy = np.matmul(intrinsic, modified_xyzs).T
    zs = xy[:,2:]
    xy = xy[:,0:2] / zs
    xy = np.round(xy)
    xy = xy.astype(np.int32)

    in_img = (xy[:,0]>=0) * (xy[:,0]<w) * (xy[:,1]>=0) * (xy[:,1]<h) * (zs[:,0]>0)

    # print("{}, has {} points to be projected".format('_'.join(elems[9:]), in_img.sum()))

    # plot the points
    # for simplicity, Here I rounded whole points.

    proj = np.zeros((int(w),int(h),4))

    for cnt, _in in enumerate(in_img):
        if _in == 0:
            continue

        proj[xy[cnt,0],xy[cnt,1],0:3] += rgbs[cnt]


        for j in range(49):
            dx = j%5 - 3
            dy = j//5 - 3

            if xy[cnt,0]+dx >= proj.shape[0] or xy[cnt,1]+dy >= proj.shape[1]:
                continue
            if xy[cnt,0]+dx < 0 or xy[cnt,1]+dy < 0:
                continue
            

            proj[xy[cnt,0]+dx,xy[cnt,1]+dy,0:3] += rgbs[cnt]
            proj[xy[cnt,0]+dx,xy[cnt,1]+dy,3] += 1

    proj = proj + 1e-8
    proj = proj / proj[:,:,3:4]

    proj = proj[:,:,0:3]

    proj = np.transpose(proj, [1, 0, 2])

    # read images
    org_img = cv2.imread(image_path)

    # overlay images
    overlay = org_img * 0.5 + proj * 0.5

    # merge images
    tot_img = np.concatenate([org_img, proj, overlay], axis = 1)

    # save results
    res_path = osp.join(res_dir, '_'.join(elems[9:]))

    cv2.imwrite(res_path, tot_img)




def debug():
    set_ind = 1
    if set_ind == 0:
        txt_dir = '/mnt/hdd/auto_colmap/kaist_scene_2_human_1011/kaist_scene_2_only_bg/output/sparse'
        img_dir = '/mnt/hdd/auto_colmap/kaist_scene_2_human_1011/kaist_scene_2_only_bg/output/images'
        res_dir = '/home/inhee/VCL/insect_recon/vision_tools/evaluation' + '/bg_wo_mask'
    elif set_ind == 1:
        txt_dir = '/mnt/hdd/auto_colmap/kaist_scene_2_human_1009/kaist_scene_2_static_inhee_2/output/sparse'
        img_dir = '/mnt/hdd/auto_colmap/kaist_scene_2_human_1009/kaist_scene_2_static_inhee_2/output/images'
        res_dir = '/home/inhee/VCL/insect_recon/vision_tools/evaluation' + '/static_wo_mask_ray'

    #projector(txt_dir, img_dir, res_dir)
    ray_projector(txt_dir, img_dir, res_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_dir', required=True, type=str, help='text file directory')
    parser.add_argument('--img_dir', required=True, type=str, help='image file directory')
    parser.add_argument('--res_dir', required=True, type=str, help='result directory')

    parser.add_argument('--use_ray', action='store_true', help='use ray or not')

    opt = parser.parse_args()

    if opt.use_ray:
        print("use ray parallel for pose evaluation")
        ray_projector(opt.txt_dir, opt.img_dir, opt.res_dir)
    else:
        projector(opt.txt_dir, opt.img_dir, opt.res_dir)




if __name__ == "__main__":
    main()
