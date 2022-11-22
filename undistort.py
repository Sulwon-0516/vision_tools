'''
Python Code for undistort from specific cameras
Here, I made a code for "Gopro".
'''

import numpy as np
import cv2
from pathlib import Path, PurePath
from tqdm import tqdm


class GoproUNndistort():
    def __init__(self):
        self.measures = [
            dict(
                intrinsic= np.array([
                    [1769.60561310104, 0, 1927.08704019384],
                    [0, 1763.89532833387, 1064.40054933721],
                    [0.,             0.,                1.]
                ]),
                radialDistortion = np.array([
                    [-0.244052127306437],
                    [0.0597008096110524],        #assumed no tangential distortion here. 
                    [0],
                    [0]
                ])
            )
        ]

    def n_measures(self):
        '''return the # of measurements'''
        return len(self.measures)    
    
    def undistort(self, img, ver=0):
        '''
        undistort images
        args :
        - img : cv2 format. (np array)
        - ver : (default = 0) the version of intrinsic / distortion measurements
        return :
        unditorted images (with cropping out black-area)
        '''
        mtx = self.measures[ver]['intrinsic']
        dist = self.measures[ver]['radialDistortion']

        # get cameras
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        cropped_dst = dst[y:y+h, x:x+w]

        return dst, cropped_dst



def debug_gopro():
    print("start Testing!")
    out = Path('/home/inhee/VCL/insect_recon/vision_tools/test') / 'gopro_undistort'
    out.mkdir(parents=True, exist_ok=True)

    # load test imgs
    src_dir = Path('/mnt/hdd/auto_colmap/kaist_scene_2_human/_raw/kaist_scene_2_dynamic_inhee_1/raw_images')
    test_img_paths = src_dir.glob('*.png')
    test_imgs = []

    for im_path in test_img_paths:
        test_imgs.append({'img':cv2.imread(str(PurePath(im_path))), 'path':im_path})

    print("start undistorting!")
    # get undistortred results
    UD = GoproUNndistort()
    for test_img in tqdm(test_imgs):
        res, cropped_res = UD.undistort(test_img['img'])
        comp = np.concatenate((test_img['img'], res), axis=0)
        cv2.imwrite(str(out/test_img['path'].name), comp)
        cv2.imwrite(str(out/('cropped_'+str(test_img['path'].name))), cropped_res)
    
    print("Testing finished!")



if __name__ == '__main__':
    debug_gopro()
