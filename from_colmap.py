
'''
This is code loading camera coordinates from COLMAP.

Currently, SIMPLE_RADIAL is defined here.

Currently, only handling "shared camera" (08/01 inhee)
'''
import numpy as np
import os
import os.path as osp

ENFORCE_PINHOLE = False



def colmap_load(pjpath):
    '''
    pjpath: project path
    '''
    c_file = osp.join(pjpath, 'cameras.txt')
    im_file = osp.join(pjpath, 'images.txt')


    K, H, W = load_intrinsic(c_file)
    RTs, img_ids, img_fnames = load_camera(im_file)

    pj_cameras = {
        'K': K,
        'H': H,
        'W': W,
        'RTs': RTs,
        'image_ids': img_ids,
        'image_fnames': img_fnames
    }

    return pj_cameras


def load_camera(im_file):
    '''
    im_file: image position annotated file
    '''
    with open(im_file, 'r') as f:
        lines = f.readlines()
    n_view = int((len(lines) - 4)/2)

    img_ids = []
    img_fnames = []
    for i in range(n_view):
        c_raw = lines[4 + 2*i].split(" ")
        img_id = int(c_raw[0])
        img_fname = c_raw[9]
        img_fname = img_fname[0:-1]

        img_ids.append(img_id)
        img_fnames.append(img_fname)
        # quaternion
        QW = float(c_raw[1])
        QX = float(c_raw[2])
        QY = float(c_raw[3])
        QZ = float(c_raw[4])
        # translation
        TX = float(c_raw[5])
        TY = float(c_raw[6])
        TZ = float(c_raw[7])

        # ref : http://www.songho.ca/opengl/gl_quaternion.html#:~:text=Multiplying%20Quaternions%20implies%20a%20rotation,cheaper%20than%20the%20matrix%20multiplication.
        RT = np.array(
            [[
                [1-2*QY*QY-2*QZ*QZ, 2*QX*QY-2*QW*QZ, 2*QX*QZ+2*QW*QY, TX],
                [2*QX*QY+2*QW*QZ, 1-2*QX*QX-2*QZ*QZ, 2*QY*QZ-2*QW*QX, TY],
                [2*QX*QZ-2*QW*QY, 2*QY*QZ+2*QW*QX, 1-2*QX*QX-2*QY*QY, TZ],
            ]]
        )

        if i == 0:
            RTs = RT
        else:
            RTs = np.concatenate((RTs, RT), axis=0)
        

    return RTs, img_ids, img_fnames




def load_intrinsic(c_file):
    '''
    c_file: camera file
    '''

    with open(c_file, 'r') as f:
        lines = f.readlines()
    
    n_camera = len(lines) - 3
    if n_camera > 1:
        print("currently only shared camera can be loaded")
        assert(0)
    

    c_raw = lines[3].split(" ")
    c_type = c_raw[1]

    # load intrinsics
    if c_type == 'SIMPLE_PINHOLE':
        K, H, W = simple_pinhole(c_raw[2:])
    elif c_type == 'SIMPLE_RADIAL':
        if ENFORCE_PINHOLE:
            K, H, W = simple_pinhole(c_raw[2:])
        else:
            K, H, W = simple_radial(c_raw[2:])
    else:
        print("currently <", c_type, "> is not supported")
        assert(0)


    return K, H, W
    


def simple_pinhole(param):
    if len(param) < (3+2):
        print("not enough # of param")
        assert(0)
    
    W = int(param[0])
    H = int(param[1])

    f = float(param[2])

    cx = float(param[3])
    cy = float(param[4])

    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ]
    )

    return K, W, H


def simple_radial(param):
    if len(param) < (3+3):
        print("not enough # of param")
        assert(0)
    
    W = int(param[0])
    H = int(param[1])

    f = float(param[2])

    cx = float(param[3])
    cy = float(param[4])
    skew_c = float(param[5])

    K = np.array(
        [
            [f, skew_c, cx],
            [0, f, cy],
            [0, 0, 1]
        ]
    )

    return K, W, H



if __name__ == '__main__':
    TEST_PATH1 = '/mnt/hdd/colmap_projects/stag_pj6_pinhole_less_constrain'
    TEST_PATH2 = '/mnt/hdd/colmap_projects/stag_pj3'


    print("test PINHOLE")
    print(colmap_load(TEST_PATH1))


    print("test RADIAL")
    print(colmap_load(TEST_PATH2))