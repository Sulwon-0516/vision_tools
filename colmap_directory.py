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
import subprocess
from tqdm import tqdm

# my files
from scene_segment import dir_segment, select_predictor
from undistort import GoproUNndistort

'''some fixed settings'''
MIN_IMG = 10            # minimum required image, for single colmap reconstruction

ENFORCE_SINGLE_MODEL = True                             # colmap mapper, enforcing single model
BMD_DIR = "/neuman/preprocess/BoostingMonocularDepth"   # Boost Monocular Depth estimation (fixed directory)


def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trim_fps_mode', action='store_true', help='Use fps first mode. default is fixing # of frame')
    parser.add_argument('--trim_fps', type=int, default=2, help='fps of the sequencer')
    parser.add_argument('--trim_skip_n_sec', default=0, type=int, help='skip first n-second when sequencing')
    parser.add_argument('--trim_N', type=int, default=150, help='# of images per each videos')


    parser.add_argument('--video', default=0, type=int, help='0:trimmed input, 1:video, 2:already processed data (used video path as input)')
    parser.add_argument('--path', required=True, type=str, help='the path to the source video')
    parser.add_argument('--out_path', default='/mnt/hdd/auto_colmap', type=str, help="the master path of colmap output")
    parser.add_argument('--sequential_matching', default=False, type=bool, help='use sequential matching instead of exhaustive during colmap')
    parser.add_argument('--pre_masking', action='store_true', help='whether apply masking before colmap')
    parser.add_argument('--resize', default=1, type=int, help='the resize ratio')

    parser.add_argument('--galaxy', action='store_true', help="Videos from galaxy has issues of upside-down. we need to rotate 180deg")
    parser.add_argument('--gopro', action='store_true', help='Apply Gopro and use pinhole as default')


    parser.add_argument('--depth', action='store_true', help='Do monocular depth estimation or not')
    parser.add_argument('--optical_flow', action='store_true', help='Extract optical flow or not')
    parser.add_argument('--only_precolmap', action='store_true', help='Only process pre-colmap')
    parser.add_argument('--skip_precolmap', action='store_true', help="skip precolmap")
    parser.add_argument('--skip_colmap', action='store_true', help="skip colmap")
    parser.add_argument('--skip_dense', action='store_true', help="skip colmap dense")
    parser.add_argument("--colmap_threads", default=-1, type=int, help='# of threads of colmap')
    parser.add_argument('--use_radial', action='store_true', help="use raidal camera for feature extraction")
    parser.add_argument('--use_opencv', action='store_true', help="use opencv camera for feature extraction")
    parser.add_argument('--use_pinhole', action='store_true', help="use opencv camera for feature extraction")


    parser.add_argument('--to_nerf', action='store_true', help="Change format into NeRF shape")

    parser.add_argument('--pose_eval', action='store_true', help="apply pose evaluation")

    parser.add_argument('--project_title', default="", type=str, help='additional project title')
    parser.add_argument('--ngp_eval', action='store_true', help='apply ngp testing')


    opt = parser.parse_args()

    # initialize mask predictor
    m_name = 'COCO_RCNN_50'
    predictor, cfg = select_predictor(m_name)

    if opt.skip_colmap:
        opt.skip_precolmap = True
    
    # updated in 22.11.22 (we apply manual undistortion on gopro, so use pinhole model)
    if opt.gopro:
        opt.use_pinhole = True

        opt.use_radial = False
        opt.use_opencv = False

    # updated in 22.10.28 (to change use_radial as default)
    if opt.use_opencv:
        opt.use_radial = False
        opt.use_pinhole = False
    elif opt.use_pinhole:
        opt.use_opencv = False
        opt.use_radial = False
    else:
        # the default setting is "radial"
        # If duplicated, the order of importance is 
        # opencv -> pinhole -> radial
        opt.use_radial = True
        opt.use_opencv = False
        opt.use_pinhole = False

    # initialize the vid2img settings
    SKIP_SECONDS = opt.trim_skip_n_sec
    GOAL_IMGS = opt.trim_N

    print("--------------------------------------------------")
    print("Path validity checking start")
    
    # check whether the input path is valid.
    if not os.path.isdir(opt.path):
        print("'{}' is not existing directory".format(opt.path))
        assert()

    # check whether there exists videos / folders of images.
    if opt.video == 1:
        # it should hold, one of following extension
        # TODO: add more extensions
        VIDEO_EXT = [
            "MP4", "mp4", "AVI", "avi", "MOV", "mov"
        ]
        video_list = []
        for ext in VIDEO_EXT:
            v_list = glob.glob(osp.join(opt.path, '*.'+ext))
            video_list.extend(v_list)
        
        video_name = []
        if len(video_list) == 0:
            print("There is no videos to process")
            assert()
        else:
            video_name = [osp.basename(v).split(".")[0] for v in video_list]
    
    else:
        # it should hold folders of images
        IMG_EXT = [
            "JPG", "jpg", "JPEG", "jpeg", "PNG", "png", "TIFF", "tiff"
        ]
        onlydirs = [osp.join(opt.path, f) for f in os.listdir(opt.path) if osp.isdir(osp.join(opt.path, f))]

        dir_list = []
        dir_name = []
        for _dir in onlydirs:
            i_list = [f for f in os.listdir(_dir) if osp.isfile(osp.join(_dir, f))]
            if len(i_list) > MIN_IMG:
                dir_list.append(_dir)
                dir_name.append(osp.basename(_dir))
        
        if len(dir_list) == 0:
            print("There is no directory containing enough images to run SfM")
            assert()


    # Check output path.
    if not os.path.isdir(opt.out_path):
        print("There's no directory '{}', so made it.".format(opt.out_path))
        os.makedirs(opt.out_path)

    
    print("Path validity checking finished")
    if opt.gopro:
        print("--------------------------------------------------")
        print("Selected --gopro options!")
        print("Discard original images & subsitute with undistorted images")
        Undistorter = GoproUNndistort()
    print("--------------------------------------------------")
    print("Video to images started")

    # Let's make result directory.
    project_title = osp.basename(opt.path) + opt.project_title
    pj_dir = osp.join(opt.out_path, project_title)

    if opt.skip_precolmap:
        if not os.path.isdir(osp.join(pj_dir, '_raw')):
            print("no processed folder for pre_colmap")
            assert()
        print("already processed. skipping pre-colmap process")
    else:
        # We will hold raw data in /_raw directory.
        os.makedirs(osp.join(pj_dir, '_raw'), exist_ok=True)

        if opt.video == 1:
            print("video to images")
            for i, video in tqdm(enumerate(video_list)):
                v_name = video_name[i]
                os.makedirs(osp.join(pj_dir ,'_raw' ,v_name), exist_ok=True)
                save_to = osp.join(pj_dir ,'_raw' ,v_name, 'raw_images')
                os.makedirs(save_to, exist_ok=True)

                # process videos.
                video_cap = cv2.VideoCapture(video)
                length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(video_cap.get(cv2.CAP_PROP_FPS))

                # calculate skipping frames
                SKIP_FRAMES = SKIP_SECONDS * fps

                # calculate required frames
                if opt.trim_fps_mode:
                    # when we use trimming based on fps.
                    every_k = opt.trim_fps
                else:
                    # when we fix total # of frames.
                    every_k = int((length-SKIP_FRAMES) // GOAL_IMGS)

                saved = 0
                for i in tqdm(range(length), desc='Processing Frames'):
                    success, image = video_cap.read()
                    if not success:
                        break
                    if i < SKIP_FRAMES:
                        continue
                    if i % every_k != 0:
                        continue

                    if opt.resize != 1:
                        w = image.shape[1] // opt.resize
                        h = image.shape[0] // opt.resize
                        image = cv2.resize(image, (w,h))

                    if opt.galaxy:
                        # we need to apply upside-down & left-right flipping.
                        image = cv2.flip(image, -1) # both side flipping == 180deg rotation

                    if opt.gopro:
                        _, image = Undistorter.undistort(image)

                    cv2.imwrite(os.path.join(save_to, f'{str(saved).zfill(5)}.png'), image)
                    saved += 1
        else:
            print("copying images")
            for i, _dir in tqdm(enumerate(dir_list)):
                dest = osp.join(pj_dir, dir_name[i])

                if opt.video==2:
                    continue
                shutil.copytree(_dir, dest, dirs_exist_ok=True)
                # If you use Python < 3.8x , use following instead shutil.copytree('bar', 'foo')

                # Currently, we should move images one more steps.
                imgs = glob.glob(osp.join(dest, '*'))
                for img in imgs:
                    shutil.move(img, osp.join(osp.abspath(img),'raw_images',osp.basename(img)))

    dir_list = [osp.join(pj_dir,'_raw',f) for f in os.listdir(osp.join(pj_dir,'_raw')) if osp.isdir(osp.join(pj_dir,'_raw',f))]
    dir_name = [osp.basename(f) for f in dir_list]
    res_list = [osp.join(pj_dir, f) for f in dir_name]

    print("Video to images finished")
    print("--------------------------------------------------")
    print("colmap pre-processing started")
    # Here I assumed all dependencies are installed in single docker container
    steps = 2+int(opt.pre_masking)+int(opt.depth)+int(opt.optical_flow)

    if not opt.skip_precolmap:
        if opt.pre_masking:
            # Generate masks
            for i, _dir in tqdm(enumerate(dir_list)):
                dir_segment(
                    predictor = predictor, 
                    cfg = cfg, 
                    _dir = osp.join(_dir,'raw_images') , 
                    save_bbox = False, 
                    bg_is_zero = False, 
                    output_path = _dir,
                )

        print("extracted pre-mask")


    if opt.only_precolmap:
        print("pre_colmap is finished")
        assert()
    
    '''
    As colmap stereo matching take tons of time, the part below will be conducted video-by-video.
    -> to leverage effects when error occurs
    '''
    for i, _dir in enumerate(dir_list):
        
        # make res directory
        res_dir = res_list[i]
        os.makedirs(res_dir, exist_ok=True)
        # make output directory
        output_dir = osp.join(res_dir, 'output')
        try:
            os.makedirs(output_dir, exist_ok=False)
        except:
            print("------------------------------------------")
            print("------------------------------------------")
            print("skip colmap processing as the previously processed results exist!")
            print("------------------------------------------")
            print("------------------------------------------")
            continue


        if opt.skip_colmap:
            print("------------------------------------------")
            print("skip colmap processing")
            print("------------------------------------------")
        else:

            rimg_dir = osp.join(_dir, 'raw_images')
            
            # make colmap dir
            colmap_dir = osp.join(res_dir, 'colmap')
            os.makedirs(colmap_dir, exist_ok=True)
            # make sparse directory
            sparse_dir = osp.join(colmap_dir, 'sparse')
            os.makedirs(sparse_dir, exist_ok=False)     # (10/07) currently, repeated execusion will make error
            # make dense directory
            dense_dir = osp.join(colmap_dir, 'dense')
            os.makedirs(dense_dir, exist_ok=False)     # (10/07) currently, repeated execusion will make error

            print("RUN COLMAP on ", dir_name[i])
            # run colmap feature_extractor
            # WARNING : you should add single spacing at the end of argument
            fe_cmd = "colmap feature_extractor "
            fe_cmd += "--database_path {} ".format(osp.join(colmap_dir, 'db.db'))
            fe_cmd += "--image_path {} ".format(rimg_dir)
            if opt.pre_masking:
                fe_cmd += "--ImageReader.mask_path {} ".format(osp.join(_dir, 'masks'))
            fe_cmd += "--SiftExtraction.estimate_affine_shape=true "
            fe_cmd += "--SiftExtraction.domain_size_pool=true "
            if opt.use_radial:
                fe_cmd += "--ImageReader.camera_model SIMPLE_RADIAL "
            elif opt.use_pinhole:
                fe_cmd += "--ImageReader.camera_model SIMPLE_PINHOLE "
            else:
                fe_cmd += "--ImageReader.camera_model OPENCV "
            fe_cmd += "--ImageReader.single_camera 1 "

            if opt.colmap_threads != -1:
                fe_cmd += "--SiftExtraction.num_threads {} ".format(opt.colmap_threads)
            do_system(fe_cmd)

            # run colmap feature matching
            if not opt.sequential_matching:
                # default mode. exhaustive matching
                fm_cmd = "colmap exhaustive_matcher "
            else:
                fm_cmd = "colmap sequential_matcher " 
            fm_cmd += "--database_path {} ".format(osp.join(colmap_dir, 'db.db'))
            fm_cmd += "--SiftMatching.guided_matching=true "
            if opt.colmap_threads != -1:
                fe_cmd += "--SiftMatching.num_threads {} ".format(opt.colmap_threads)
            do_system(fm_cmd)

            # run reconstruction
            map_cmd = "colmap mapper "
            map_cmd += "--database_path {} ".format(osp.join(colmap_dir, 'db.db'))
            map_cmd += "--image_path {} ".format(rimg_dir)
            map_cmd += "--output_path {} ".format(sparse_dir)
            if ENFORCE_SINGLE_MODEL:
                map_cmd += "--Mapper.multiple_models 0 " # default allows multipled model
            if False:
                '''add reconstruction options here'''
                map_cmd += "--Mapper.init_num_trials {} ".format(400) # double inititalize trial for robust recon. default 200
                map_cmd += "--Mapper.multiple_models 0 " # default allows multipled model
            
            if opt.colmap_threads != -1:
                map_cmd += "--Mapper.num_threads {} ".format(opt.colmap_threads)
            do_system(map_cmd)

            # check result
            if not ENFORCE_SINGLE_MODEL:
                if len(os.listdir(sparse_dir))!= 1:
                    print("Bad reconstruction")
                    assert()        ######## You can keep running the code with commenting out this part
                else:
                    print("with muliple-model, all images are allocated")
            
            # run Bundle Adjustment
            ba_cmd = "colmap bundle_adjuster "
            ba_cmd += "--BundleAdjustment.refine_principal_point 1 "
            sparse_dir = osp.join(sparse_dir,"0")
            ba_cmd += "--input_path {} ".format(sparse_dir)
            ba_cmd += "--output_path {} ".format(sparse_dir)
            do_system(ba_cmd)

            # undistort images
            ud_cmd = "colmap image_undistorter "
            ud_cmd += "--image_path {} ".format(rimg_dir)
            ud_cmd += "--input_path {} ".format(sparse_dir)
            ud_cmd += "--output_path {} ".format(dense_dir)
            do_system(ud_cmd)

            if not opt.skip_dense:
                # stereo matching
                sm_cmd = "colmap patch_match_stereo "
                sm_cmd += "--workspace_path {} ".format(dense_dir)
                do_system(sm_cmd)

            # convert to txt
            save_cmd = "colmap model_converter "
            save_cmd += "--input_path {} ".format(osp.join(dense_dir, 'sparse'))
            save_cmd += "--output_path {} ".format(osp.join(dense_dir, 'sparse'))
            save_cmd += "--output_type=TXT "
            do_system(save_cmd)

            # copy results
            dest = osp.join(output_dir, 'images')
            shutil.copytree(osp.join(dense_dir, 'images'), dest, dirs_exist_ok=True)
            if not opt.skip_dense:
                dest = osp.join(output_dir, 'depth_maps')
                shutil.copytree(osp.join(dense_dir, 'stereo', 'depth_maps'), dest, dirs_exist_ok=True)
            dest = osp.join(output_dir, 'sparse')
            shutil.copytree(osp.join(dense_dir, 'sparse'), dest, dirs_exist_ok=True)

            
            # get masks for rectified images
            print("RUN DETECTRON2 on ", dir_name[i])
            dir_segment(
                predictor = predictor, 
                cfg = cfg, 
                _dir = osp.join(output_dir,'images') , 
                save_bbox = True, 
                bg_is_zero = False, 
                output_path = osp.join(output_dir, 'segmentations'),
            )

        # Process the data into NeRF shape
        if opt.to_nerf:
            nerf_dir = osp.join(res_dir,'NeRF_format')
            os.makedirs(nerf_dir, exist_ok=True)
            c2n_cmd = "python /home/inhee/VCL/insect_recon/vision_tools/colmap2nerf.py "
            c2n_cmd += "--images {} ".format(osp.join(output_dir, 'images'))
            c2n_cmd += " --text {} ".format(osp.join(output_dir, 'sparse'))
            c2n_cmd += "--split True "
            c2n_cmd += "--test_ratio 0.1 "
            c2n_cmd += "--val_ratio 0.1 "
            c2n_cmd += "--out {} ".format(nerf_dir)

            do_system(c2n_cmd)

        # if apply colmap evalution
        if opt.pose_eval:
            print("apply pose evaluation")

            eval_txt_dir = osp.join(output_dir, 'sparse')
            eval_img_dir = osp.join(output_dir, 'images')
            eval_res_dir = osp.join(output_dir, 'pose_eval')

            eval_cmd = "python /home/inhee/VCL/insect_recon/vision_tools/colmap_evaluation.py "
            eval_cmd += "--txt_dir {} ".format(eval_txt_dir)
            eval_cmd += "--img_dir {} ".format(eval_img_dir)
            eval_cmd += "--res_dir {} ".format(eval_res_dir)

            # we default use ray parallel to process it quickly
            eval_cmd += "--use_ray "

            do_system(eval_cmd)

        # if apply ngp evaluation
        if opt.ngp_eval:    
            print("apply simple ngp evaluations")
            if not opt.to_nerf:
                print("we need to do nerf processing first")

        # Monocular depth estimation
        if opt.depth:
            print("RUN Monocular depth estimation on ", dir_name[i])
            cwd = os.getcwd()
            os.chdir(BMD_DIR)
            
            # Check settings
            if not os.path.exists(os.path.join(BMD_DIR, 'pix2pix/checkpoints/mergemodel')):
                os.makedirs(os.path.join(BMD_DIR, 'pix2pix/checkpoints/mergemodel'))
            if not os.path.isfile(os.path.join(BMD_DIR, 'pix2pix/checkpoints/mergemodel/latest_net_G.pth')):
                do_system('wget https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth -O {}'.format(osp.join(BMD_DIR, "pix2pix/checkpoints/mergemodel/latest_net_G.pth")))
            if not os.path.isfile(os.path.join(BMD_DIR, 'res101.pth')):
                do_system('wget https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download -O res101.pth')

            # Run monocular depth estimation
            if not os.path.isdir(os.path.join(output_dir, 'mono_depth')):
                bmd_cmd = '. ~/.bashrc && conda activate BMD '        # TODO : if it doesn't work, substitue it with absolute path in docker
                bmd_cmd += '&& python run.py --Final --data_dir {} --output_dir {} --depthNet 2 '.format(osp.join(output_dir, "images"), osp.join(output_dir, "mono_depth"))
                bmd_cmd += '&& conda deactivate'
                print("run following code \n", bmd_cmd)
                subprocess.run(bmd_cmd, shell=True)         # TODO : Check it works well with shell=True option, which means running on independent shell
            else:
                print("it already has monocular depth estimation results")
                assert()
            os.chdir(cwd)

        # Optical Flow
        if opt.optical_flow:
            print("(11/22) not prepared yet")
            assert()
    


if __name__ == "__main__":
    main()
