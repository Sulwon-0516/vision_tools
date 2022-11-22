if [ $# -eq 1 ]
    then
        GPU_ID="1"
    else
        GPU_ID=$2
fi


if [ $1 == "1" ]
    then 
        echo ": default preprocess with resolution modification"
        
        python colmap_directory.py \
            --video 1 \
            --path /mnt/hdd/videos/20221005_dataset/kaist_scene_2_human \
            --out_path /mnt/hdd/auto_colmap/kaist_scene_2_human_1080 \
            --pre_masking \
            --depth \
            --skip_dense \
            --to_nerf \
            --resize 2 

elif [ $1 == "2" ]
    then 
        echo ": default preprocess"
        
        python colmap_directory.py \
            --video 1 \
            --path /mnt/hdd/videos/20221022_dataset \
            --out_path /mnt/hdd/auto_colmap/20221022_dataset \
            --skip_dense \
            --to_nerf \
            --pose_eval

elif [ $1 == "3" ]
    then 
        echo ": default preprocess with radial"
        
        python colmap_directory.py \
            --video 1 \
            --path /mnt/hdd/videos/20221023_dataset \
            --out_path /mnt/hdd/auto_colmap \
            --skip_dense \
            --use_radial \
            --skip_precolmap \
            --to_nerf


elif [ $1 == "4" ]
    then 
        echo $1": default preprocess with radial / 2022.11.21 quick mobile processing"
        
        python colmap_directory.py \
            --video 1 \
            --path /mnt/hdd/videos/20221122_dataset_mobile \
            --out_path /mnt/hdd/auto_colmap \
            --pre_masking \
            --depth \
            --use_radial \
            --to_nerf

elif [ $1 == "5" ]
    then 
        echo $1": default preprocess with radial / 2022.11.22 with flippings "
        
        python colmap_directory.py \
            --video 1 \
            --path /mnt/hdd/videos/20221122_dataset_mobile \
            --out_path /mnt/hdd/auto_colmap \
            --pre_masking \
            --depth \
            --use_radial \
            --to_nerf \
            --galaxy \
            --project_title _flipped

elif [ $1 == "6" ]
    then 
        echo $1": default preprocess / 2022.11.22 with gopro preprocessing "
        
        python colmap_directory.py \
            --video 1 \
            --path /mnt/hdd/videos/20221122_dataset_gopro \
            --out_path /mnt/hdd/auto_colmap \
            --pre_masking \
            --depth \
            --to_nerf \
            --gopro \
            --project_title _preundist

fi