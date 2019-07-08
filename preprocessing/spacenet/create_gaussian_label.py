#!/usr/bin/env python2

"""
create_gaussian_label.py: script to convert Spacenet linestring annotation to gaussian road mask.

It will create following directory structure:
    base_dir
        | ---> gaussian_roads
                    | ---> label_tif : Tiff image to raster Linestring as road skeleton image.
                    | ---> label_png : PNG image to create gaussian road mask.
"""

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import cv2
from scipy.ndimage.morphology import *
import glob
import math
import time
from osgeo import gdal
import geoTools as gT
from tqdm import tqdm
tqdm.monitor_interval = 0


def CreateGaussianLabel(base_dir):
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

        geojson_dir = os.path.join(base_dir,'{country}/geojson/spacenetroads/'.format(country=country).format(country=country))
        rgb_dir = os.path.join(base_dir,'{country}/RGB-PanSharpen/'.format(country=country))

        roads_dir = os.path.join(base_dir,'{country}/gaussian_roads'.format(country=country))

        if os.path.isdir(roads_dir) == False:
            os.makedirs(roads_dir)
            os.makedirs(os.path.join(roads_dir,'label_tif'))
            os.makedirs(os.path.join(roads_dir,'label_png'))


        ## The default image size of Spacenet Dataset is 1300x1300.
        black_image = np.zeros((1300,1300),dtype=np.uint8)

        failure_count = 0
        index = 0
        print('Processing Images from {}'.format(country))
        print('*'*60)

        progress_bar = tqdm(glob.glob(geojson_dir + '/*.geojson'), ncols=150)
        for file_ in progress_bar:
            name = file_.split('/')[-1]
            index += 1
            
            file_name = name.replace('spacenetroads_','').replace('.geojson','')
            progress_bar.set_description("  | --> Creating: {}".format(file_name))
            
            geojson_name_format = 'spacenetroads_{0}.geojson'.format(file_name)
            rgb_name_format = 'RGB-PanSharpen_{0}.tif'.format(file_name)
            road_segment_name_format = 'RGB-PanSharpen_{0}.tif'.format(file_name)
            out_tif_file = os.path.join(roads_dir,'label_tif',road_segment_name_format)
            out_png_file = os.path.join(roads_dir,'label_png',road_segment_name_format).replace('.tif','.png')
            
            geojson_file = os.path.join(geojson_dir,geojson_name_format)
            tif_file = os.path.join(rgb_dir,rgb_name_format)
            
            status = gT.ConvertToRoadSegmentation(tif_file,geojson_file,out_tif_file)
            
            if status != 0:
                print("|xxx-> Not able to convert the file {}. <-xxx".format(name))
                failure_count += 1
                cv2.imwrite(out_png_file,black_image)
            else:
                gt_dataset = gdal.Open(out_tif_file, gdal.GA_ReadOnly)
                if not gt_dataset:
                    continue
                gt_array = gt_dataset.GetRasterBand(1).ReadAsArray()
                
                distance_array = distance_transform_edt(1-(gt_array/255))
                std = 15
                distance_array =  np.exp(-0.5*(distance_array*distance_array)/(std*std))
                distance_array *= 255
                cv2.imwrite(out_png_file,distance_array)

            # print('\t|--> Processed Images : {}'.format(index), end='\r')
            # sys.stdout.flush()
            # print('\t|--> Image: {}'.format(file_name))

        print("Not able to convert {} files.".format(failure_count))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_dir', type=str, required=True, 
        help='Base directory for Spacenent Dataset.')

    args = parser.parse_args()

    start = time.clock()
    CreateGaussianLabel(args.base_dir)
    end = time.clock()
    print('Finished Creating Labels, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()