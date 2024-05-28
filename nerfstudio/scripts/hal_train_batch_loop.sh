AUGMENT_STEP=('1000' '5000' '8000' '9000' '10000' '11000' '12000' '15000')
name=${NAME:-augment_step}


for ((i=0;i<${#AUGMENT_STEP[@]};i++)); do
    sbatch nerfstudio/scripts/halnew.sh $name --pipeline.augment-phase-step  ${AUGMENT_STEP[$i]}
done