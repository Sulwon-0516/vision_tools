from genericpath import exists
import tensorflow as tf
import os
import os.path as osp
import json
import tqdm
import cv2
import numpy as np

TRAIN_RES_PATH = "/mnt/hdd/dataset/blocknerf/train"
BLOCK_NERF_PATH="/mnt/hdd/dataset/blocknerf/v1.0"
files = tf.io.gfile.glob(osp.join(BLOCK_NERF_PATH, 'waymo_block_nerf_mission_bay_train.tfrecord*'))
print(len(files))
files=files[11:]
dataset = tf.data.TFRecordDataset(filenames=files, compression_type='GZIP')

iterator = dataset.as_numpy_iterator()

os.makedirs(TRAIN_RES_PATH, exist_ok=True)
os.makedirs(osp.join(TRAIN_RES_PATH, "spec"), exist_ok=True)
os.makedirs(osp.join(TRAIN_RES_PATH, "img"), exist_ok=True)


cnt = 0
for entry in iterator:
    print(cnt)
    example = tf.train.Example()
    example.MergeFromString(entry)

    result = {}
    # example.features.feature is the dictionary
    for key, feature in example.features.feature.items():
        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list

        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value).tolist()

    # Decode RGB image.
    image_hash = result['image_hash'][0]
    del result['image']
    img_data = example.features.feature['image'].bytes_list.value[0]

    json_name = osp.join(TRAIN_RES_PATH, "spec",str(image_hash)+".json")
    img_name = osp.join(TRAIN_RES_PATH, "img",str(image_hash)+".png")


    with open(json_name, 'w') as fp:
        json.dump(result, fp)


    rgb = tf.image.decode_png(img_data).numpy()
    cv2.imwrite(img_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


    
    # it's only contained in validation sets
    #print(example.features.feature['mask'])

    cnt += 1