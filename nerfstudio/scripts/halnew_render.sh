#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH --output /staging/agp/masterthesis/nerf-thesis-shared/test/%A_%a.out
#SBATCH --array=1
#SBATCH --job-name=neurad_shift
#SBATCH --partition zprodlow


configpath=${CONFIGPATH}
if [ -z ${configpath} ]; then
    echo "configpath not set. Exiting."
    exit 1;
fi

dataset=${DATASET:-pandaset}
outputpath=${OUTPUTPATH:-renders}
subcommand=${SUBCOMMAND:-dataset}
image_path=${IMAGE_PATH:-"/staging/agp/masterthesis/nerf-thesis-shared/containers/neuraddiffusion-24_05_24.sif"}


if [ "$dataset" == "zod" ]; then
    dataset_root="/staging/dataset_donation/round_2"
elif [ "$dataset" == "nuscenes" ]; then
    dataset_root="/datasets/nuscenes/v1.0"
elif [ "$dataset" == "pandaset" ]; then
    dataset_root="/staging/agp/datasets/pandaset"
else
    echo "Dataset must be either zod or nuscenes or pandaset, got $dataset"
    exit 1
fi

singularity exec --nv \
    --bind $PWD:/nerfstudio \
    --bind /staging:/staging \
    --bind /workspaces:/workspaces \
    --bind /datasets:/datasets \
    --env WANDB_API_KEY=$wandb_api_key \
    $image_path \
    python3.10 -u nerfstudio/scripts/render.py \
    $subcommand \
    --load-config $configpath \
    --output-path $outputpath \

#
#EOF