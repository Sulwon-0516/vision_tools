'''
Here I build a trajectoy generation file
'''

# TODO: scale & axis system of nerf is somewhat strange.... we need to test it with raw colmap data.
 
import argparse
import math
import numpy as np
import os
import os.path as osp
import commentjson as json

def parse_args():
    parser = argparse.ArgumentParser(description="generate automate camera trajectory for torch-ngp")
    parser.add_argument("--json_path", default="/mnt/hdd/auto_colmap/kaist_scene_2_human_1009/kaist_scene_2_static_inhee_2/NeRF_format/transforms_train.json", type=str, help="transformed json path to load.")
    parser.add_argument("--sparse_path", default="/mnt/hdd/auto_colmap/kaist_scene_2_human_1009/kaist_scene_2_static_inhee_2/output/sparse", type=str, help="undistorted sparse results")
    parser.add_argument("--out_path", default="/home/inhee/VCL/insect_recon/instant-ngp/render_path/colmap2traj.json")
    parser.add_argument("--num_up", default=2, type=int, help="cnt of upper way move")
    args = parser.parse_args()

    return args


if False:
    import quaternion
    def get_scale(frames):
        # as we normalized the position, we can extract the scale of scene from min / max values
        mins = [1000, 1000, 1000]
        maxs = [-1000, -1000, -1000]
        for frame in frames:
            x = frame['transform_matrix'][0][3]
            y = frame['transform_matrix'][1][3]
            z = frame['transform_matrix'][2][3]

            if x > maxs[0]:
                maxs[0] = x
            elif x < mins[0]:
                mins[0] = x

            if y > maxs[1]:
                maxs[1] = y
            elif y < mins[1]:
                mins[1] = y

            if z > maxs[2]:
                maxs[2] = z
            elif z < mins[2]:
                mins[2] = z

            
        # select minimum value from x,y
        scales = [abs(mins[0]), abs(mins[1]), maxs[0], maxs[1]]
        scale = min(scales)

        return scale


    def get_cameras(json_dir):
        '''
        Extract frames from transformation.json file and sort them
        '''
        if not os.path.exists(json_dir):
            print("no such file:, {}".format(json_dir))
            assert()

        with open(json_dir, "r") as f:
            raw_json = json.load(f)

        frames = raw_json['frames']

        # we need to align it by file name.
        for i, frame in enumerate(frames):
            frame['order'] = int(osp.basename(frame['file_path']).split(".")[0])
            frames[i] = frame

        # algin the file list as following
        frames = sorted(frames, key=lambda frame: frame['order'])

        return frames
        

    def gen_trajectory(frames, num_up, out_path):
        '''
        generate trajectory of frames.
        args:
        - num_up : # of upper move in camera trajectory.
        '''
        n_points = 16        # resolution of camera path

        n_frames = len(frames)
        frame_interval = n_frames // n_points
        scale = get_scale(frames)

        height = scale * 0.8 * 3       # 0.8 is manually selected number
        
        final_cameras = []

        for i in range(n_points):
            final_cameras.append(frames[i*frame_interval]['transform_matrix'])
        

        # let's make smooth move
        '''
        ind = 0
        for i in range(n_points-1):
            cam = frames[i*frame_interval]['transform_matrix']
            cam[1][3] += height * abs(math.sin(math.pi*i*num_up/(n_points-1)))
            final_cameras.append(cam)
        final_cameras.append(frames[(n_points-1)*frame_interval]['transform_matrix'])
        '''
        
        # change shapes as following
        # it's only applied on instant-ngp
        final_json = {'path':[]}

        for cam in final_cameras:
            np_cam = np.array(cam)
            rot_matrix = np_cam[0:3,0:3]
            q = quaternion.from_rotation_matrix(rot_matrix)
            t = np_cam[0:3,3]
            t = -t[[0,2,1]]

            t_scale = math.sqrt(sum(abs(t)**2))
            t = t / t_scale
            print(q)

            # now make new lists
            cam_dict = dict()
            cam_dict['R'] = quaternion.as_float_array(q)
            cam_dict['R'] = cam_dict['R'][[0,1,3,2]].tolist()

            cam_dict['T'] = t.tolist()
            cam_dict['aperture_size'] = 0.0
            cam_dict['fov'] = 50.625        # here I fixed the value
            cam_dict['scale'] = t_scale
            cam_dict['slice'] = 0.0

            final_json['path'].append(cam_dict)

        # write the results in the directory
        with open(out_path, "w") as f:
            json.dump(final_json, f)


def colmap2video(sparse_dir, out_path, N_cam = 10):
    SKIP_EARLY = 0
    with open(os.path.join(sparse_dir,"images.txt"), "r") as f:
        i = 0
        frames = []
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
                frame = dict()
                frame['R'] = np.array(tuple(map(float, elems[1:5])))
                frame['T'] = np.array(tuple(map(float, elems[5:8])))
                frame['order'] = int(elems[-1].split(".")[0])
                frames.append(frame)

    frames = sorted(frames, key=lambda frame: frame['order'])
    f_int = len(frames) // N_cam

    inds = [i*f_int for i in range(N_cam)]
    frames = [frames[ind] for ind in inds]


    # change it into ngp trajectory
    # it's only applied on instant-ngp
    final_json = {'path':[]}

    for frame in frames:
        t = frame['T'][[0,2,1]]
        t_scale = math.sqrt(sum(abs(t)**2))
        t = t / t_scale

        # now make new lists
        cam_dict = dict()
        cam_dict['R'] = frame['R'][[0,1,3,2]].tolist()
        cam_dict['T'] = t.tolist()
        cam_dict['aperture_size'] = 0.0
        cam_dict['fov'] = 50.625        # here I fixed the value
        cam_dict['scale'] = t_scale
        cam_dict['slice'] = 0.0

        final_json['path'].append(cam_dict)

    # write the results in the directory
    with open(out_path, "w") as f:
        json.dump(final_json, f)




if __name__ == '__main__':
    opt = parse_args()
    colmap2video(opt.sparse_path, opt.out_path)
    '''
    frames = get_cameras(opt.json_path)
    gen_trajectory(frames, opt.num_up,  opt.out_path)
    '''
