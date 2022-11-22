
import argparse
import sys
import os
import os.path as osp
import glob
import shutil
import cv2
import subprocess
from tqdm import tqdm


def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)



def colmap2nerf():

	# we need to first process following paths
    paths_to_process = [
		'/mnt/hdd/dataset/nerf_benchmark/nerf_format/360_v2',
		'/mnt/hdd/dataset/nerf_benchmark/nerf_format/nerf_real_360'
	]