: '
Bash Script file to Prepare Spacenet Images and Gaussian Road Masks.
1) Convert Spacenet 11-bit images to 8-bit Images, country wise.
2) Create Gaussian Road Masks, country wise.
3) Move all data to single folder.
'

spacenet_base_dir=$1

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'
BOLD='\033[1m'
UNDERLINE='\033[4m'

printf "${BLUE}${BOLD}Spacenet Data Base Dir => %s \n${NC}" $spacenet_base_dir

printf "${GREEN}${UNDERLINE}${BOLD}\n Converting Spacenet 11-bit RGB images to 8-bit. ${NC}\n"
python spacenet/convert_to_8bit_png.py -d $spacenet_base_dir

printf "${GREEN}${UNDERLINE}${BOLD}\n Creating Spacenet gaussian road labels. ${NC}\n"
python spacenet/create_gaussian_label.py -d $spacenet_base_dir

printf "${GREEN}${UNDERLINE}${BOLD}\n Copying data to $spacenet_base_dir/full. ${NC}\n"
for dir in $(find $spacenet_base_dir -maxdepth 1 -type d)
	do
		image_folder="$dir/RGB_8bit"
		copy_star="/*"
		if [ -d "$image_folder" ]; then
			mkdir -p "$spacenet_base_dir/full/images/"
			# mkdir -p "$spacenet_base_dir/full/labels/"
		    cp $image_folder$copy_star "$spacenet_base_dir/full/images/"
		fi
		label_folder="$dir/gaussian_roads/label_png"
		if [ -d "$label_folder" ]; then
			mkdir -p "$spacenet_base_dir/full/gt/"
		    cp $label_folder$copy_star "$spacenet_base_dir/full/gt/"
		fi
	done






