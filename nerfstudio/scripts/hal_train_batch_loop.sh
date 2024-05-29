AUGMENT_STEP=('1000' '5000' '8000' '9000' '10000' '11000' '12000' '15000')
name=${NAME:-augment_step}


for ((i=0;i<${#AUGMENT_STEP[@]};i++)); do
    sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  ${AUGMENT_STEP[$i]}
done

# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  1000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  3000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  5000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  8000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  9000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  10000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  11000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  12000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  14000
# sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  15000

