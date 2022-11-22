'''
visual hull implemented results
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import cv2
from from_colmap import colmap_load
from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from utils import *
import imageio


RES_DIR = '/home/inhee/VCL/results/visual_hull'
EXP_NAME = 'debug_4_0.50_coarse_5x5x5'
CAMERA_DIR = '/mnt/hdd/colmap_projects/docker_stag_test_pj1/after_BA'
SIL_DIR = '/mnt/hdd/gen_dataset/video_Han_Stag/thrs_mask'
TEST_SET_DIR = '/home/inhee/VCL/results/visual_hull/VH_debug_set/set_1'
USE_CPU = False

os.makedirs(RES_DIR, exist_ok=True)

THRS = 0.5  # voxelize threshold
ang_interv = 6
ANGLES_phi = [a * ang_interv for a in range(0, 180//ang_interv)]
ANGLES_theta = [a * ang_interv for a in range(0, 180//ang_interv)]
DPI = 200

# Here I simply assumes unit cube
class VisualHull():
    def __init__(self, device, N_grid=(50, 50, 50) , side_len=5.):
        '''
        input:
            N_grid: [N_x, N_y, N_z] 
            vox2pix: (voxel side length)/(pixel side length)
        '''
        # define voxel grid.
        # I'll define the "Origin" at the center of voxel-field.
        # Easily, (-N_x/2 ~ N_x/2-1)

        self.device = device
        self.N_grids = N_grid
        self.vox_size = side_len / N_grid[0]

        self.vox_cnt = torch.zeros(size=self.N_grids, dtype=torch.uint8).to(device)     # (#) of rays passing voxel
        self.vox_fov = torch.zeros(size=self.N_grids, dtype=torch.uint8).to(device)     # (#) of bg/fg ray passing voxel


        # define voxel coordinates
        x = torch.arange(start=0, end=self.N_grids[0]) - (self.N_grids[0]-1)/2.
        y = torch.arange(start=0, end=self.N_grids[1]) - (self.N_grids[1]-1)/2.
        z = torch.arange(start=0, end=self.N_grids[2]) - (self.N_grids[2]-1)/2.

        xx = x.reshape(1,-1,1,1).repeat(1, 1, self.N_grids[1], self.N_grids[2])
        yy = y.reshape(1,1,-1,1).repeat(1, self.N_grids[0], 1, self.N_grids[2])
        zz = z.reshape(1,1,1,-1).repeat(1, self.N_grids[0], self.N_grids[1], 1)

        vox_coord = torch.cat([xx, yy, zz], dim=0)
        self.vox_coord = vox_coord.to(device)        # (N_x, N_y, N_z, 3) it's integer or +- 0.5 (need rescale)
        self.r_vox_coord = self.vox_coord * self.vox_size


    def multi_camera(self, K, Rs, Ts, sil_paths, Xmax, Ymax):
        '''
        sil_paths : list of paths including silhouettes
        '''
        print("===============================")
        print("start optimization")
        print("N camera : ", len(sil_paths))
        print("===============================")    

        Rs = Rs.to(self.device)
        Ts = Ts.to(self.device)
        K = K.to(self.device)

        for i, sil_path in tqdm(enumerate(sil_paths)):
            sil_image = cv2.imread(sil_path)
            sil_image = sil_image.squeeze()
            
            if (len(sil_image.shape) != 3) and (len(sil_image.shape) != 2):
                print("strange silhouette shape")
                assert(0)
            
            elif (len(sil_image.shape) == 3):
                if sil_image.shape[-1]==4:
                    # handle RGBA
                    sil_image = sil_image[...,3]
                else:
                    sil_image = sil_image[...,0]
            
            sil_image = sil_image // sil_image.max()
            silhouette = torch.tensor(sil_image, dtype=torch.uint8).to(self.device)

            R = Rs[i]
            T = Ts[i]

            self.single_camera(
                K = K,
                R = R,
                T = T,
                silhouette = silhouette,
                Xmax = Xmax,
                Ymax = Ymax
            )

        print("===============================")
        print("Finished optimization ")
        print("===============================")




    def single_camera(self, K, R, T, silhouette, Xmax, Ymax):
        '''
        input:
        - R : roataion matrix
        - T : translation matrix
        - silhouette : silhouette (0/1, torch tensor), (X,Y) ordered (assume)
        - Xmax : max X (longer one)
        - Ymax : max Y
        '''

        # world -> camera
        _, Nx, Ny, Nz = self.r_vox_coord.shape
        temp = self.r_vox_coord.reshape(3, -1)
        coords = R @ temp + T.unsqueeze(-1)
        

        # camera -> img plane
        coords = K @ coords

        coords = coords.reshape(3, Nx, Ny, Nz)

        z_coords = coords[2]    # it should be positive!
        # homogenous -> inhomogenous
        coords = coords / coords[2:,:,:,:]

        # round pixels (kinda Nearest Neighbor)
        coords = torch.round(coords)

        x_coord = coords[0]     # (N_x, N_y, N_z)
        y_coord = coords[1]     # (N_x, N_y, N_z)

        # if (x,y) is in image_plane        
        in_img = ( (((x_coord >= 0) * (x_coord < Xmax)) * ((y_coord >= 0) * (y_coord < Ymax))) * (z_coords>0))
        # add to vox_fov
        self.vox_fov += in_img

        # if (x,y) is in silhouette
        

        # extend temporaly
        ext_x_coord = (in_img*x_coord).reshape(-1)
        ext_y_coord = (in_img*y_coord).reshape(-1)
        sum_coord = ext_x_coord * Ymax + ext_y_coord

        # find coord
        in_sil = torch.index_select(silhouette.reshape(-1), 0, sum_coord.type(torch.IntTensor).to(self.device))
        in_sil = in_sil.reshape(Nx, Ny, Nz) * in_img

        self.vox_cnt += in_sil

    
    def check_reproj(self, voxel, path, K, R, T, silhouette, Xmax, Ymax):
        '''
        Check whether visual hull is working well with re-projection
        '''

        # world -> camera
        _, Nx, Ny, Nz = self.r_vox_coord.shape
        temp = self.r_vox_coord.reshape(3, -1)
        coords = R @ temp + T.unsqueeze(-1)
        

        # camera -> img plane
        coords = K @ coords

        coords = coords.reshape(3, Nx, Ny, Nz)

        z_coords = coords[2]    # it should be positive!
        # homogenous -> inhomogenous
        coords = coords / coords[2:,:,:,:]

        # round pixels (kinda Nearest Neighbor)
        coords = torch.round(coords)

        x_coord = coords[0]     # (N_x, N_y, N_z)
        y_coord = coords[1]     # (N_x, N_y, N_z)

        # if (x,y) is in image_plane        
        valids = ( (((x_coord >= 0) * (x_coord < Xmax)) * ((y_coord >= 0) * (y_coord < Ymax))) * (z_coords>0)) * voxel
        # add to vox_fov

        pix_lists = coords.reshape(3,-1)[0:2] * valids.reshape(1,-1)
        pix_lists = pix_lists.type(torch.IntTensor)


        
        canvas = np.zeros((Xmax, Ymax), dtype=np.uint8)
        for pix in pix_lists.T:
            canvas[pix[0], pix[1]] = 1

        silhouette = silhouette.cpu().detach().numpy()


        my_subplots([silhouette, canvas], 2, 2, ("Visual_hull_reporj: "+EXP_NAME), ["input","reporj"])
        plt.savefig(path, dpi = 300, transparent = False)

        


    def gen_voxels(self, thrs_ratio = 0.2):
        '''
        thrs_ratio : if (in silhouette) / (visualized) < thrs_ratio : it's empty
        '''
        
        ratio = self.vox_cnt / self.vox_fov
        voxel = (ratio > thrs_ratio)
        voxel = voxel.cpu().numpy().astype(bool)

        return voxel

    def save_res(self, path):
        '''
        input:
        - path : it should be individual folders (per each experiments)
        '''
        os.makedirs(path, exist_ok=True)

        np.save(osp.join(path, 'vox_cnt.npy'), self.vox_cnt.cpu().numpy())
        np.save(osp.join(path, 'vox_fov.npy'), self.vox_fov.cpu().numpy())
        
        print("===============================")
        print('[vox_cnt.npy] and [vox_fov.npy] are saved in : ', path)
        print("===============================")



def single_vox_visualizer(voxel, res_path, title = EXP_NAME):
    # currently w/o cameras
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.voxels(filled=voxel)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    os.makedirs(res_path, exist_ok=True)
    img_path = osp.join(res_path, 'imgs')
    os.makedirs(img_path, exist_ok=True)
        
    n_frame = len(ANGLES_phi) if len(ANGLES_phi) < len(ANGLES_theta) else len(ANGLES_theta)

    filenames = []
    for i in tqdm(range(len(ANGLES_theta))):
        ax.view_init(10, ANGLES_theta[i])
        fname = EXP_NAME + '_2_' + str(i) + '.png'
        fname = os.path.join(img_path, fname)
        plt.savefig(fname, transparent = False, dpi=DPI)
        filenames.append(fname)

    for i in tqdm(range(len(ANGLES_phi))):
        ax.view_init(ANGLES_phi[i], ANGLES_theta[-1])
        fname = EXP_NAME + '_1_' + str(i) + '.png'
        fname = os.path.join(img_path, fname)
        plt.savefig(fname, transparent = False, dpi=DPI)
        filenames.append(fname)

    
    for i in tqdm(range(n_frame)):
        ax.view_init(ANGLES_phi[-1-i], ANGLES_theta[-1-i])
        fname = EXP_NAME + '_3_' + str(i) + '.png'
        fname = os.path.join(img_path, fname)
        plt.savefig(fname, transparent = False, dpi=DPI)
        filenames.append(fname)

    with imageio.get_writer(res_path+'/'+EXP_NAME+'_res.gif', mode='I') as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)



def main():
    # device setting
    if not USE_CPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # generate path
    res_path = osp.join(RES_DIR, EXP_NAME)
    os.makedirs(res_path, exist_ok=True)

    # intializes
    VH = VisualHull(device=device)
    colmap_pose = colmap_load(CAMERA_DIR)

    # settings
    Rs = torch.from_numpy(colmap_pose['RTs'][:,0:3,0:3].astype(np.float32))
    Ts = torch.from_numpy(colmap_pose['RTs'][:,0:3,3].astype(np.float32))
    image_fnames = [osp.join(SIL_DIR, fname) for fname in colmap_pose['image_fnames']]

    # optimize
    VH.multi_camera(
        K = torch.from_numpy(colmap_pose['K'].astype(np.float32)),
        Rs = Rs,
        Ts = Ts,
        sil_paths = image_fnames,
        Xmax = colmap_pose['W'],
        Ymax = colmap_pose['H'] ################################## ERRONEOUS
    )
    VH.save_res(osp.join(res_path, 'save'))

    # visualize res
    voxels = VH.gen_voxels(thrs_ratio=THRS)
    single_vox_visualizer(voxels, res_path, EXP_NAME+'_%.2f'%(THRS))



def debug():
    # device setting
    if not USE_CPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # generate path
    res_path = osp.join(RES_DIR, EXP_NAME)
    os.makedirs(res_path, exist_ok=True)

    # intializes
    VH = VisualHull(device=device)
    colmap_pose = colmap_load(CAMERA_DIR)

    # settings
    Ks = np.load(osp.join(TEST_SET_DIR, 'Ks.npy')).astype(np.float32)
    Rs = np.load(osp.join(TEST_SET_DIR, 'Rs.npy')).astype(np.float32)
    Ts = np.load(osp.join(TEST_SET_DIR, 'Ts.npy')).astype(np.float32)
    image_fnames = [osp.join(TEST_SET_DIR, 'imgs', 'image_%d.png'%(i)) for i in range(100)]

    # optimize
    VH.multi_camera(
        K = torch.from_numpy(Ks),
        Rs = torch.from_numpy(Rs),
        Ts = torch.from_numpy(Ts),
        sil_paths = image_fnames,
        Xmax = 256,
        Ymax = 256 ################################## ERRONEOUS
    )
    VH.save_res(osp.join(res_path, 'save'))

    # visualize res
    voxels = VH.gen_voxels(thrs_ratio=THRS)
    single_vox_visualizer(voxels, res_path, EXP_NAME+'_%.2f'%(THRS))



    # check reporj
    Rs = torch.from_numpy(Rs).to(device)
    Ts = torch.from_numpy(Ts).to(device)
    K = torch.from_numpy(Ks).to(device)

    # reproj path
    reproj_path = osp.join(res_path, 'reproj')
    os.makedirs(reproj_path, exist_ok=True)


    for i, sil_path in tqdm(enumerate(image_fnames)):
        rep_fname = osp.join(reproj_path, 'image_%d.png'%(i))
        sil_image = cv2.imread(sil_path)
        sil_image = sil_image.squeeze()
        
        if (len(sil_image.shape) != 3) and (len(sil_image.shape) != 2):
            print("strange silhouette shape")
            assert(0)
        
        elif (len(sil_image.shape) == 3):
            if sil_image.shape[-1]==4:
                # handle RGBA
                sil_image = sil_image[...,3]
            else:
                sil_image = sil_image[...,0]
        
        sil_image = sil_image // sil_image.max()
        silhouette = torch.tensor(sil_image, dtype=torch.uint8).to(device)

        R = Rs[i]
        T = Ts[i]

        VH.check_reproj(
            voxel=torch.from_numpy(voxels).to(device),
            path=rep_fname,
            K = K,
            R = R,
            T = T,
            silhouette = silhouette,
            Xmax = 256,
            Ymax = 256
        )

    print("===============================")
    print("Finished optimization ")
    print("===============================")



if __name__=='__main__':
    with torch.no_grad():
        debug()


