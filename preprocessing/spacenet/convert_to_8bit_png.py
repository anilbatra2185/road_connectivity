#!/usr/bin/env python2

"""
create_png.py: script to convert Spacenet 11-bit RGB images to 8-bit images and 
preprocessing with CLAHE (a variant of adaptive histogram equalization algorithm).

It will create following directory structure:
    base_dir
        | ---> RGB_8bit : Save 8-bit png images.
"""

from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import cv2
import glob
import tifffile as tif
import time
from tqdm import tqdm
tqdm.monitor_interval = 0


def CreatePNG(base_dir):
	spacenet_countries = ['AOI_2_Vegas_Roads_Train',
							'AOI_3_Paris_Roads_Train',
							'AOI_4_Shanghai_Roads_Train',
							'AOI_5_Khartoum_Roads_Train']

	for country in spacenet_countries:
	    tif_folder = os.path.join(base_dir,'{country}/RGB-PanSharpen/'.format(country=country))
	    if os.path.isdir(tif_folder) == False:
			print(" !  RGB-PanSharpen folder does not exist for {country}.  !  ".format(country=country))
			print('x'*80)
			continue

	    out_png_dir = os.path.join(base_dir,'{country}/RGB_8bit'.format(country=country))

	    if os.path.isdir(out_png_dir) == False:
	        os.makedirs(out_png_dir)


	    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
	    print('Processing Images from {}'.format(country))
	    print('*'*80)

	    progress_bar = tqdm(glob.glob(tif_folder + '/*.tif'), ncols=150)
	    for file_ in progress_bar:

	        file_name = file_.split('/')[-1].replace('.tif','.png')
	        progress_bar.set_description("  | --> Converting: {}".format(file_name))

	        img=tif.imread(file_)
	        red = np.asarray(img[:,:,0],dtype=np.float)
	        green = np.asarray(img[:,:,1],dtype=np.float)
	        blue = np.asarray(img[:,:,2],dtype=np.float)

	        red_ = 255.0 * ((red-np.min(red))/(np.max(red) - np.min(red)))
	        green_ = 255.0 * ((green-np.min(green))/(np.max(green) - np.min(green)))
	        blue_ = 255.0 * ((blue-np.min(blue))/(np.max(blue) - np.min(blue)))

	        ## The default image size of Spacenet Dataset is 1300x1300.
	        img_rgb = np.zeros((1300,1300,3),dtype=np.uint8)
	        img_rgb[:,:,0] = clahe.apply(np.asarray(red_,dtype=np.uint8))
	        img_rgb[:,:,1] = clahe.apply(np.asarray(green_,dtype=np.uint8))
	        img_rgb[:,:,2] = clahe.apply(np.asarray(blue_,dtype=np.uint8))

	        cv2.imwrite(os.path.join(out_png_dir,file_name),img_rgb[:,:,::-1])
	        
	        # print('\t|--> Processed Images : {}'.format(index), end='\r')
	        # time.sleep(1)
            # sys.stdout.flush()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--base_dir', type=str, required=True, 
		help='Base directory for Spacenent Dataset.')

	args = parser.parse_args()

	start = time.clock()
	CreatePNG(args.base_dir)
	end = time.clock()

	print('Finished Creating png, time {0}s'.format(end - start))

if __name__ == "__main__":
	main()