#!/usr/bin/env python3

'''
Inhee modified little bit (added test/train split with adding images)

'''
# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

'''
09.22
This colmap2nerf file is modifeid to specific on Kaist_scene_1

'''

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import copy
import os
import shutil

def parse_args():
	parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

	parser.add_argument("--video_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")
	parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL","OPENCV"], help="camera model")
	parser.add_argument("--colmap_camera_params", default="", help="intrinsic parameters, depending on the chosen model.  Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images", help="input path to the images")
	parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
	parser.add_argument("--aabb_scale", default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
	parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
	parser.add_argument("--set", default=-1, choices=["1","2","3","4","-1"], help="the name of set")
	parser.add_argument("--select", default=-1, help="select following amount of images")
	parser.add_argument("--split", default=True, help="split image with random order")
	parser.add_argument("--test_ratio", default=0.1, help="ratio of test images")
	parser.add_argument("--val_ratio", default=0.1, help="ratio of validation images")
	parser.add_argument("--keep_colmap_coords", action="store_true", help="keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")
	parser.add_argument("--out", default="transforms.json", help="output path")
	parser.add_argument("--vocab_path", default="", help="vocabulary tree path")
	parser.add_argument("--scale_position", default=1.0, help="scale parameter compared to default colmap-2-nerf process")

	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

def run_ffmpeg(args):
	if not os.path.isabs(args.images):
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)
	images = "\"" + args.images + "\""
	video =  "\"" + args.video_in + "\""
	fps = float(args.video_fps) or 1.0
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
	if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	try:
		# Passing Images' Path Without Double Quotes
		shutil.rmtree(args.images)
	except:
		pass
	do_system(f"mkdir {images}")

	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"
	do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")

def run_colmap(args):
	db = args.colmap_db
	images = "\"" + args.images + "\""
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=args.text
	sparse=db_noext+"_sparse"
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
	if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
	do_system(f"colmap feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
	match_cmd = f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
	if args.vocab_path:
		match_cmd += f" --VocabTreeMatching.vocab_tree_path {args.vocab_path}"
	do_system(match_cmd)
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

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

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

if __name__ == "__main__":
    args = parse_args()
    if args.video_in != "":
        run_ffmpeg(args)
    if args.run_colmap:
        run_colmap(args)
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images
    TEXT_FOLDER = args.text
    OUT_PATH = args.out
    os.makedirs(OUT_PATH, exist_ok=True)
    os.chdir(OUT_PATH)
    
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY*2:
                continue
                    

            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b=sharpness(name)
                print(i, ",", name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not args.keep_colmap_coords:
                    c2w[0:3,2] *= -1 # flip the y and z axis
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:] # swap y and z
                    c2w[2,:] *= -1 # flip whole world upside down

                    up += c2w[0:3,1]

                frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
                out["frames"].append(frame)
    nframes = len(out["frames"])
    

    if args.keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.01:
                    totp += p*w
                    totw += w
        totp /= totw
        print(totp) # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 * float(args.scale_position) / avglen # scale to "nerf sized"

    for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()

    
    # from here, it's all after applying transformation.
    # In other word, the offset / scaling factor doesn't affect below here.
    # select specific # of images
    if int(args.set) == -1:
        pass
    elif int(args.set) == 1:
        out['frames'] = out['frames'][0:32]
        nframes = len(out['frames'])
        
    elif int(args.set) == 2:
        out['frames'] = out['frames'][32:402]
        nframes = len(out['frames'])

    elif int(args.set) == 3:
        out['frames'] = out['frames'][402:846]
        nframes = len(out['frames'])

    elif int(args.set) == 4:
        out['frames'] = out['frames'][846:]
        nframes = len(out['frames'])
    else:
        assert(0)
        print("unexpected error")



    if args.split:
        if args.select != -1:
            ind_list = np.arange(nframes)
            np.random.shuffle(ind_list)
            if nframes > int(args.select):
                ind_list, _ = np.split(ind_list, [int(args.select)])
            ind_lists = np.split(ind_list, [int(.8 * len(ind_list)), int(.9 * len(ind_list))])
        else:
            ind_list = np.arange(nframes)
            np.random.shuffle(ind_list)
            ind_lists = np.split(ind_list, [int(.8 * len(ind_list)), int(.9 * len(ind_list))])

        # make dicts
        out_train = copy.deepcopy(out)
        out_test = copy.deepcopy(out)
        out_val = copy.deepcopy(out)

        # make image dir
        os.makedirs(os.path.join(OUT_PATH, 'train'), exist_ok=True)
        os.makedirs(os.path.join(OUT_PATH, 'test'), exist_ok=True)
        os.makedirs(os.path.join(OUT_PATH, 'val'), exist_ok=True)

        # modify frames
        out_train['frames'] = [out_train['frames'][ind] for ind in ind_lists[0]]
        out_test['frames'] = [out_test['frames'][ind] for ind in ind_lists[1]]
        out_val['frames'] = [out_val['frames'][ind] for ind in ind_lists[2]]

        # move files
        for i in range(len(out_train['frames'])):
            frame = out_train['frames'][i]
            f_n = os.path.basename(frame['file_path'])
            new_path = os.path.join(OUT_PATH, 'train', f_n)
            shutil.copyfile(frame['file_path'], new_path)

            rel_path = os.path.relpath(new_path)
            out_train['frames'][i]['file_path']=str(f"./{rel_path}")

        for i in range(len(out_test['frames'])):
            frame = out_test['frames'][i]
            f_n = os.path.basename(frame['file_path'])
            new_path = os.path.join(OUT_PATH, 'test', f_n)
            shutil.copyfile(frame['file_path'], new_path)

            rel_path = os.path.relpath(new_path)
            out_test['frames'][i]['file_path']=str(f"./{rel_path}")
        
        for i in range(len(out_val['frames'])):
            frame = out_val['frames'][i]
            f_n = os.path.basename(frame['file_path'])
            new_path = os.path.join(OUT_PATH, 'val', f_n)
            shutil.copyfile(frame['file_path'], new_path)

            rel_path = os.path.relpath(new_path)
            out_val['frames'][i]['file_path']=str(f"./{rel_path}")


        print("train/test/val: %d/%d/%d"%(len(ind_lists[0]), len(ind_lists[1]), len(ind_lists[2])))
        print(f"writing {OUT_PATH}")

        with open(os.path.join(OUT_PATH, "transforms_train.json"), "w")as outfile:
            json.dump(out_train, outfile, indent=2)

        with open(os.path.join(OUT_PATH, "transforms_test.json"), "w") as outfile:
            json.dump(out_test, outfile, indent=2)

        with open(os.path.join(OUT_PATH, "transforms_validation.json"), "w") as outfile:
            json.dump(out_val, outfile, indent=2)



    else:
        if args.select != -1:
            ind_list = np.arange(nframes)
            np.random.shuffle(ind_list)
            if nframes > int(args.select):
                ind_list, _ = np.split(ind_list, [int(args.select)])
            new_frames = [out['frames'][ind] for ind in ind_list]
            out['frames'] = new_frames
        print(nframes,"frames")
        print(f"writing {OUT_PATH}")
        with open(os.path.join(OUT_PATH, "transforms.json"), "w") as outfile:
            json.dump(out, outfile, indent=2)