'''
preprocessing code to apply pifu / pifuhd in videos
'''
import numpy as np
import os.path as osp
import os
import glob
import cv2
from math import floor, ceil

'''deprecated'''
def old_crop_img(master_path, res_path, img_name = '00030.png', resize=512):
    # load images
    img_path = osp.join(master_path, 'NeRF_format', 'train', img_name)
    img = cv2.imread(img_path)

    # load bbox informations
    bbox_path = osp.join(master_path, 'output', 'segmentations', 'bbox', img_name, '*.npy')
    bbox_list = glob.glob(bbox_path)


    for i, bbox in enumerate(bbox_list):
        print(bbox)
        bbox_np = np.load(bbox)[0]

        x1 = int(floor(bbox_np[0]))
        x2 = int(floor(bbox_np[3]))
        y1 = int(ceil(bbox_np[1]))
        y2 = int(ceil(bbox_np[2]))

        w = x2-x1
        h = y2-y1

        w = w if w>h else h
        h = w

        x = floor((x1+x2 - w)/2)
        y = floor((y1+y2 - h)/2)

        '''Crop in rectangular shape'''
        '''pad imgs when bbox is out of img'''
        x_front = 0   # offset for the case when we padded in front of the img.
        y_front = 0
        x_back = 0
        y_back = 0

        # load masks
        mask_path = osp.join(osp.dirname(bbox), osp.basename(bbox).split(".")[0]+'.png')
        mask = cv2.imread(mask_path)
        
        if x<0:
            x_front = -x
        if y<0:
            y_front = -y
        if x+w>= img.shape[1]:
            x_back = x+w-img.shape[1]+1
        if y+h>=img.shape[0]:
            y_back = y+w-img.shape[0]+1

        if x_front+y_front+x_back+y_back > 0:
            ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_REPLICATE)
            ext_mask = cv2.copyMakeBorder(mask, y_front, y_back, x_front, x_back, cv2.BORDER_REPLICATE)
            x = x + x_front
            y = y + y_front
        else:
            ext_img = img
            ext_mask = mask

        cropped_img = ext_img[y:y+h, x:x+h, :]
        cropped_mask = ext_mask[y:y+h, x:x+h, :]

        cv2.imwrite(osp.join(res_path, str(i).zfill(5)+'_'+img_name), cropped_img)
        cv2.imwrite(osp.join(res_path, 'mask_'+str(i).zfill(5)+'_'+img_name), cropped_mask)

        

def crop_img(img_path, res_path, bbox_path, mask_path, resize=512):
    # load images
    img = cv2.imread(img_path)
    img_name = osp.basename(img_path).split(".")[0]

    # load bbox informations
    # print(bbox_path)
    bbox_np = np.load(bbox_path)[0]

    x1 = int(floor(bbox_np[0]))
    x2 = int(floor(bbox_np[2]))
    y1 = int(ceil(bbox_np[1]))
    y2 = int(ceil(bbox_np[3]))

    w = int((x2-x1) * 1.2)
    h = int((y2-y1) * 1.2)

    #print(h, "  ", w)

    w = w if w>h else h
    h = w

    x = floor((x1+x2 - w)/2)
    y = floor((y1+y2 - h)/2)

    '''Crop in rectangular shape'''
    '''pad imgs when bbox is out of img'''
    x_front = 0   # offset for the case when we padded in front of the img.
    y_front = 0
    x_back = 0
    y_back = 0

    # load masks
    mask = cv2.imread(mask_path)
    
    if x<0:
        x_front = -x
    if y<0:
        y_front = -y
    if x+w>= img.shape[1]:
        x_back = x+w-img.shape[1]+1
    if y+h>=img.shape[0]:
        y_back = y+w-img.shape[0]+1

    if x_front+y_front+x_back+y_back > 0:
        print(img_name, " is extended\n")
        ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_REPLICATE)
        ext_mask = cv2.copyMakeBorder(mask, y_front, y_back, x_front, x_back, cv2.BORDER_REPLICATE)
        x = x + x_front
        y = y + y_front
    else:
        ext_img = img
        ext_mask = mask

    cropped_img = ext_img[y:y+h, x:x+h, :]
    cropped_mask = ext_mask[y:y+h, x:x+h, :]

    cv2.imwrite(osp.join(res_path, img_name+'.png'), cropped_img)
    cv2.imwrite(osp.join(res_path, img_name+'_mask.png'), cropped_mask)



def debug_crop():
    CROP_DEBUG_DIR = '/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic'
    RES_PATH = '/home/inhee/VCL/insect_recon/vision_tools/test/20221026_dataset_iphone_statue'

    os.makedirs(RES_PATH, exist_ok=True)
    

    crop_img(CROP_DEBUG_DIR, RES_PATH)



def pre_crop():
    SELECTED_LIST = '/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected'

    npy_list = glob.glob(osp.join(SELECTED_LIST, "*.npy"))

    res_path = osp.join(osp.dirname(SELECTED_LIST), 'cropped')
    os.makedirs(res_path, exist_ok=True)

    for i, npy in enumerate(npy_list):
        img_name = osp.basename(npy).split(".")[0]+'.png'
        mask_path = osp.join(SELECTED_LIST, img_name)
        bbox_path = npy
        img_path = osp.join(osp.dirname(osp.dirname(SELECTED_LIST)), 'images', img_name)
        
        crop_img(img_path, res_path, bbox_path, mask_path, resize=512)
        




if __name__=='__main__':
    pre_crop()