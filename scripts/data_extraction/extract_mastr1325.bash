#!/bin/bash
input_dir="data/raw/mastr1325/" 
output_dir="data/processed/mastr1325/"

imgs_file="MaSTr1325_images_512x384.zip"
masks_file="MaSTr1325_masks_512x384.zip"
imus_file="MaSTr1325_imus_512x384.zip"

folder_names=("images" "masks" "imus")
zip_files=($imgs_file $masks_file $imus_file)

mkdir -p $output_dir

for i in ${!zip_files[@]}; do
    mkdir -p ${output_dir}/${folder_names[$i]}
    echo "Extracting ${zip_files[$i]} from $input_dir into $output_dir/${folder_names[$i]}"
    unzip -o $input_dir${zip_files[$i]} -d $output_dir/${folder_names[$i]}
    echo "Done"
done