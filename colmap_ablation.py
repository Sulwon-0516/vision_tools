'''
It's ablation study of colmap and it's performance.
Please refer to following slides regarding the results.

slide: https://docs.google.com/presentation/d/1vaFI1vEQgh8r-w5_cqqiDoOfZwAB6QaR-nySpyeqqqA/edit?usp=sharing

'''

import sys
import os
import os.path as osp
import shutil

TESTING_VIDEOS = [
    "/mnt/hdd/videos/20221005_dataset/kaist_scene_2_human/kaist_scene_2_only_bg.MP4",
    "/mnt/hdd/videos/20221005_dataset/kaist_scene_2_human/kaist_scene_2_static_inhee_2.MP4"
]



def test_1028():
    mpath='/mnt/hdd/videos/20221026_dataset'

    code_1_1 = "python colmap_directory.py "
    code_1_1 += "--video 1 "
    code_1_1 += "--path {} ".format(mpath)
    code_1_1 += "--project_title {} ".format("_iphone_statue")
    code_1_1 += "--pre_masking "
    code_1_1 += "--to_nerf "
    code_1_1 += "--use_radial "
    code_1_1 += "--resize {} ".format("1")
    code_1_1 += "--pose_eval "
    code_1_1 += "--colmap_threads {} ".format("8")
    code_1_1 += "--depth "

    do_system(code_1_1)


def test_1022():
    mpath='/mnt/hdd/auto_colmap/20221022_dataset/20221022_dataset'
    dirs = os.listdir(mpath)

    nerf_dirs = []
    save_dirs = []
    for _dir in dirs:
        if _dir == '_raw':
            continue
        nerf_dir = osp.join(mpath, _dir, 'NeRF_format')
        nerf_dirs.append(osp.join(nerf_dir, 'transforms_train.json'))
        save_dirs.append(osp.join(mpath, _dir, 'output', 'nerf_test'))
    
    #train_ngp(nerf_dirs, save_dirs)
    to_video_1022(save_dirs)

def test():
    a,b = test_nerf()
    train_ngp(a,b)


def test_nerf():
    dirlists = [
        "/mnt/hdd/auto_colmap/colmap_ablation_test_1_4k/",
        "/mnt/hdd/auto_colmap/colmap_ablation_test_2_opencv/",
        "/mnt/hdd/auto_colmap/colmap_ablation_test_2_radial/",
        "/mnt/hdd/auto_colmap/colmap_ablation_test_3_wo_mask/"
    ]

    vid_lists = [
        "kaist_scene_2_static_inhee_2",
        "kaist_scene_2_only_bg"
    ]

    nerf_dirs = []
    save_dirs = []
    for _dir in dirlists:
        for vid in vid_lists:
            nerf_dir = osp.join(_dir, vid,'NeRF_format')
            os.makedirs(nerf_dir, exist_ok=True)
            c2n_cmd = "python /home/inhee/VCL/insect_recon/vision_tools/colmap2nerf.py "
            c2n_cmd += "--images {} ".format(osp.join(_dir, vid, 'output', 'images'))
            c2n_cmd += "--text {} ".format(osp.join(_dir, vid, 'output', 'sparse'))
            c2n_cmd += "--split True "
            c2n_cmd += "--test_ratio 0.1 "
            c2n_cmd += "--val_ratio 0.1 "
            c2n_cmd += "--out {} ".format(nerf_dir)

            #do_system(c2n_cmd)
            if _dir == "/mnt/hdd/auto_colmap/colmap_ablation_test_3_wo_mask/" and vid == "kaist_scene_2_only_bg":
                continue

            nerf_dirs.append(osp.join(nerf_dir, 'transforms_train.json'))
            save_dirs.append(osp.join(_dir, vid, 'output', 'nerf_test'))


    return nerf_dirs, save_dirs



def train_ngp(nerf_dirs, save_dirs):
    os.chdir('/home/inhee/VCL/insect_recon/instant-ngp')
    # First, train with ngp simply
    for i, nd in enumerate(nerf_dirs):
        if not osp.exists(osp.join(save_dirs[i], 'train.msgpack')):
            os.makedirs(save_dirs[i], exist_ok=True)
            train_cmd = "python scripts/run.py "
            train_cmd += "--scene {} ".format(nd)
            train_cmd += '--mode "nerf" '
            train_cmd += '--n_steps 40000 '
            train_cmd += '--save_snapshot {} '.format(osp.join(save_dirs[i], 'train.msgpack'))

            do_system(train_cmd)

        # test synthesis
        test_cmd = "python scripts/run.py "
        test_cmd += '--mode "nerf" '
        test_cmd += "--load_snapshot {} ".format(osp.join(save_dirs[i], 'train.msgpack'))
        test_cmd += "--test_transforms {} ".format(osp.join(osp.dirname(nd),'transforms_train.json'))
        test_cmd += "--video_fps 15 "
        test_cmd += "--render_path {} ".format(save_dirs[i])

        do_system(test_cmd)



def to_video_1022(save_dirs):
    for render in save_dirs:
        os.system(f'ffmpeg -y -framerate 15 -i {render}/frames/%05d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {render}/video.mp4')
        os.system(f'ffmpeg -y -framerate 15 -i {render}/diffs/%05d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {render}/diff_video.mp4')

def copy_videos():
    video_path = "/mnt/hdd/videos/colmap_ablation"
    os.makedirs(video_path, exist_ok=True)
    for video in TESTING_VIDEOS:
        shutil.copy(video, video_path)
    
    return video_path


def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)



def main():
    print("copy videos")
    video_path = copy_videos()


    print("------------------------------------------")
    print("test1, colmap by resolution (small size)")
    print("------------------------------------------")


    # test set 1 case 1 (4k)
    code_1_1 = "python colmap_directory.py "
    code_1_1 += "--video 1 "
    code_1_1 += "--path {} ".format(video_path)
    code_1_1 += "--project_title {} ".format("_test_1_4k")
    code_1_1 += "--pre_masking "
    code_1_1 += "--skip_dense "
    code_1_1 += "--to_nerf "
    code_1_1 += "--resize {} ".format("1")
    code_1_1 += "--pose_eval"

    do_system(code_1_1)

    # test set 1, case 2 (fhd)
    code_1_2 = "python colmap_directory.py "
    code_1_2 += "--video 1 "
    code_1_2 += "--path {} ".format(video_path)
    code_1_2 += "--project_title {} ".format("_test_1_fhd")
    code_1_2 += "--pre_masking "
    code_1_2 += "--skip_dense "
    code_1_2 += "--to_nerf "
    code_1_2 += "--resize {} ".format("2")
    code_1_2 += "--pose_eval"

    do_system(code_1_2)

    
    print("------------------------------------------")
    print("test2, colmap by camera (small size)")
    print("------------------------------------------")
    
    # test set 1 case 1 (opencv)
    code_2_1 = "python colmap_directory.py "
    code_2_1 += "--video 1 "
    code_2_1 += "--path {} ".format(video_path)
    code_2_1 += "--project_title {} ".format("_test_2_opencv")
    code_2_1 += "--pre_masking "
    code_2_1 += "--skip_dense "
    code_2_1 += "--to_nerf "
    code_2_1 += "--resize {} ".format("2")
    code_2_1 += "--pose_eval"

    do_system(code_2_1)

    # test set 1, case (radial)
    code_2_2 = "python colmap_directory.py "
    code_2_2 += "--video 1 "
    code_2_2 += "--path {} ".format(video_path)
    code_2_2 += "--project_title {} ".format("_test_2_radial")
    code_2_2 += "--pre_masking "
    code_2_2 += "--skip_dense "
    code_2_2 += "--to_nerf "
    code_2_2 += "--use_radial "
    code_2_2 += "--resize {} ".format("2")
    code_2_2 += "--pose_eval"

    do_system(code_2_2)


    print("------------------------------------------")
    print("test3, colmap w/o pre masking (small size)")
    print("------------------------------------------")
    
    # test set 3 case 1 (w/o masking)
    code_3_1 = "python colmap_directory.py "
    code_3_1 += "--video 1 "
    code_3_1 += "--path {} ".format(video_path)
    code_3_1 += "--project_title {} ".format("_test_3_wo_mask")
    code_3_1 += "--skip_dense "
    code_3_1 += "--to_nerf "
    code_3_1 += "--resize {} ".format("2")
    code_3_1 += "--pose_eval"

    do_system(code_3_1)

    # test set 3, case 2 == test_1_fhd 
    print("finished all process!")


if __name__ == '__main__':
    #main()
    #test()
    #test_1022()
    test_1028()