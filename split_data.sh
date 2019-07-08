# Deepglobe
# image_postfix = "_sat.jpg"
# gt_postfix = "_mask.png"

# Spacenet
# image_postfix = ".png"
# gt_postfix = ".png"

full_train_dir=$1
base_dir=$2
image_postfix=$3
gt_postfix=$4

train_image="$base_dir/train/images/"
train_gt="$base_dir/train/gt/"
val_image="$base_dir/val/images/"
val_gt="$base_dir/val/gt/"

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'
BOLD='\033[1m'
UNDERLINE='\033[4m'

printf "${BLUE}${BOLD}Full Data Dir => %s \n${NC}" $full_train_dir
printf "${BLUE}${BOLD}Split Data Dir => %s \n${NC}" $base_dir
printf "${BLUE}$Split Sub Dir => %s \n${NC}" $train_image $train_gt $val_image $val_gt

printf "${GREEN}${UNDERLINE}${BOLD}\n Creating folder structure. ${NC}\n"
mkdir -p $train_image $train_gt $val_image $val_gt

printf "${GREEN}${UNDERLINE}${BOLD}\n Splitting Data. ${NC}\n"

i=1
sp="/-\|"
echo -n ' '
while read -r line
	do
		cp "$full_train_dir/images/$line$image_postfix" "$train_image"
		cp "$full_train_dir/gt/$line$gt_postfix" "$train_gt"
		printf "\r${sp:i++%${#sp}:1} Copying training data."
	done < "$base_dir/train.txt"

i=1
while read -r line
	do
		cp "$full_train_dir/images/$line$image_postfix" "$val_image"
		cp "$full_train_dir/gt/$line$gt_postfix" "$val_gt"
		printf "\r${sp:i++%${#sp}:1} Copying validation data."
	done < "$base_dir/val.txt"

printf "\n${GREEN}${BOLD}\n Finished Split. ${NC}\n"




