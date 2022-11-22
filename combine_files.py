import glob
import os
import os.path as osp
import shutil
import cv2
import time
import sys
import json

DIR_LISTS = [
    '/mnt/hdd/trimmed_data/kaist_scene1_rot1',
    '/mnt/hdd/trimmed_data/kaist_scene1_rot2',
    '/mnt/hdd/trimmed_data/kaist_scene1_rot3',
    '/mnt/hdd/trimmed_data/kaist_scene1_rot4'
]

FINAL_DIR = "/mnt/hdd/trimmed_data/kaist_scene1"


def merge(dirs, final):
    os.makedirs(final, exist_ok=True)
    
    cnt = 1
    for _dir in dirs:
        dir_lists = sorted(glob.glob(_dir + "/*.png"), key=os.path.getmtime)

        for file in dir_lists:
            print(file)

            shutil.copy(file, os.path.join(FINAL_DIR, str(cnt).zfill(5)+".png"))
            cnt += 1
        print("--------------------------------------")


def copytree(src, dst, symlinks=False, ignore=None):
    '''
    https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    copy directory function
    '''
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)



def resize(src, ratio=2):
    dst = src + '_x'+str(ratio)
    os.makedirs(dst)
    copytree(src,dst)
    print("copied whole data")

    flists = glob.glob(osp.join(dst, "**/*.png"), recursive=True)

    for file in flists:
        img = cv2.imread(file)
        img2 = cv2.resize(img, (0, 0), fx=1/ratio, fy=1/ratio)
        print(file, " -> ", img2.shape)

        cv2.imwrite(file, img2)

    
    json_lists = glob.glob(osp.join(dst, "*.json"))

    for _json_f in json_lists:
        with open(_json_f, "r") as _json_fo:
            trans = json.load(_json_fo)

        trans['fl_x'] = trans['fl_x']/ratio
        trans['fl_y'] = trans['fl_y']/ratio

        trans['cx'] = trans['cx']/ratio
        trans['cy'] = trans['cy']/ratio
        trans['w'] = trans['w']/ratio
        trans['h'] = trans['h']/ratio

        os.remove(_json_f)
        with open(_json_f, "w") as _json_fo:
            json.dump(trans, _json_fo, indent='\t')




def resize_nerf_dataset(_dir):
    pass




if __name__ == '__main__':
    #merge(DIR_LISTS, FINAL_DIR)


    KAIST_SCENE1='/mnt/hdd/NeRF_processed_dataset/kaist_scene1_rot2_aabb16_v4'
    resize(KAIST_SCENE1,2)
    resize(KAIST_SCENE1,4)
    resize(KAIST_SCENE1,8)