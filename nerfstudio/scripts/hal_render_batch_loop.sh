config_neurad="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331981/imaginedriving-001-331981-1/imaginedriving/2024-05-25_130746/config.yml"
config_diffusion="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/331982/imaginedriving-001-331982-1/imaginedriving/2024-05-25_130744/config.yml"
config_diffusion_lora="/staging/agp/masterthesis/nerf-thesis-shared/output/neurad_imaginedriving/pandaset-imaginedriving/332043/imaginedriving-001-332043-1/imaginedriving/2024-05-26_174218/config.yml"


output_base=/staging/agp/masterthesis/nerf-thesis-shared/shifts
SHIFTS_FOLDERS=('0m' '1m' '2m' '3m' '4m' '10m')
SHIFTS=("0 0 0" "1 0 0" "2 0 0" "3 0 0" "4 0 0" "10 0 0")

for ((i=0;i<${#SHIFTS_FOLDERS[@]};i++)); do
    output_path="${output_base}/config_neurad/${SHIFTS_FOLDERS[$i]}"
    mkdir -p $output_path
    chmod -R 775 $output_path
    CONFIGPATH=$config_neurad OUTPUTPATH=${output_path} sbatch nerfstudio/scripts/halnew_render.sh --shift ${SHIFTS[$i]}
done