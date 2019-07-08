"""
The code is borrowed from Spacenet Utilities.
    https://github.com/SpaceNetChallenge/utilities/blob/spacenetV3/spacenetutilities/geoTools.py
"""

from osgeo import gdal, osr, ogr
import numpy as np
import os
import csv
import subprocess
import math
import geopandas as gpd
import shapely
from shapely.geometry import Point
from pyproj import Proj, transform
from fiona.crs import from_epsg
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
try:
    import rtree
    import centerline
    import osmnx
except:
    print("rtree not installed, Will break evaluation code")


def import_summary_geojson(geojsonfilename):
    # driver = ogr.GetDriverByName('geojson')
    datasource = ogr.Open(geojsonfilename, 0)

    layer = datasource.GetLayer()
#     print(layer.GetFeatureCount())

    imagename = geojsonfilename.split('/')[-1].replace('.geojson','.tif')
    roadlist = []
    
    ###Road Type
    #1: Motorway
    #2: Primary
    #3: Secondary
    #4: Tertiary
    #5: Residential
    #6: Unclassified
    #7: Cart track
    
    for idx, feature in enumerate(layer):

        poly = feature.GetGeometryRef()
        
        if poly:
            roadlist.append({'ImageID':imagename,'RoadID': feature.GetField('road_id'), 
                             'RoadType': feature.GetField('road_type'),
                             'Lanes': feature.GetField('lane_number'),
                             'IsBridge': feature.GetField('bridge_typ'),# 1= Bridge, 2=Not Bridge, 3=Unknown
                             'Paved': feature.GetField('bridge_typ'),# 1= Paved, 2=Unpaved, 3=Unknown
                             'LineString': feature.GetGeometryRef().Clone()})

    return roadlist

def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    # type: (object, object, object, object, object) -> object

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)

def geoWKTToPixelWKT(geom, inputRaster, targetSR, geomTransform, breakMultiGeo, pixPrecision=2):
    # Returns Pixel Coordinate List and GeoCoordinateList

    geom_list = []
    geom_pix_wkt_list = []
    
    if geom.GetGeometryName() == 'POLYGON':
        polygonPix = ogr.Geometry(ogr.wkbPolygon)
        for ring in geom:
            # GetPoint returns a tuple not a Geometry
            ringPix = ogr.Geometry(ogr.wkbLinearRing)

            for pIdx in xrange(ring.GetPointCount()):
                lon, lat, z = ring.GetPoint(pIdx)
                xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

                xPix = round(xPix, pixPrecision)
                yPix = round(yPix, pixPrecision)
                ringPix.AddPoint(xPix, yPix)

            polygonPix.AddGeometry(ringPix)
            polygonPixBuffer = polygonPix.Buffer(0.0)
            geom_list.append([polygonPixBuffer, geom])

    elif geom.GetGeometryName() == 'MULTIPOLYGON':

        for poly in geom:
            polygonPix = ogr.Geometry(ogr.wkbPolygon)
            for ring in poly:
                # GetPoint returns a tuple not a Geometry
                ringPix = ogr.Geometry(ogr.wkbLinearRing)

                for pIdx in xrange(ring.GetPointCount()):
                    lon, lat, z = ring.GetPoint(pIdx)
                    xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

                    xPix = round(xPix, pixPrecision)
                    yPix = round(yPix, pixPrecision)
                    ringPix.AddPoint(xPix, yPix)

                polygonPix.AddGeometry(ringPix)
                polygonPixBuffer = polygonPix.Buffer(0.0)
                geom_list.append([polygonPixBuffer, geom])
                
    elif geom.GetGeometryName() == 'LINESTRING':
        line = ogr.Geometry(ogr.wkbLineString)
        for pIdx in xrange(geom.GetPointCount()):
            lon, lat, z = geom.GetPoint(pIdx)
            xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

            xPix = round(xPix, pixPrecision)
            yPix = round(yPix, pixPrecision)
            line.AddPoint(xPix, yPix)
        geom_list.append([line, geom])
    
    elif geom.GetGeometryName() == 'MULTILINESTRING':
        
        if breakMultiGeo:
            for poly in geom:
                line = ogr.Geometry(ogr.wkbLineString)
                for pIdx in xrange(poly.GetPointCount()):
                    lon, lat, z = poly.GetPoint(pIdx)
                    xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

                    xPix = round(xPix, pixPrecision)
                    yPix = round(yPix, pixPrecision)
                    line.AddPoint(xPix, yPix)
                geom_list.append([line, poly])
        else:
            multiline = ogr.Geometry(ogr.wkbMultiLineString)
            for poly in geom:
                line = ogr.Geometry(ogr.wkbLineString)
                for pIdx in xrange(poly.GetPointCount()):
                    lon, lat, z = poly.GetPoint(pIdx)
                    xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

                    xPix = round(xPix, pixPrecision)
                    yPix = round(yPix, pixPrecision)
                    line.AddPoint(xPix, yPix)
                multiline.AddGeometry(line)
            geom_list.append([multiline, geom])
        
    elif geom.GetGeometryName() == 'POINT':
        point = ogr.Geometry(ogr.wkbPoint)
        for pIdx in xrange(geom.GetPointCount()):
            lon, lat, z = geom.GetPoint(pIdx)
            xPix, yPix = latlon2pixel(lat, lon, inputRaster, targetSR, geomTransform)

            xPix = round(xPix, pixPrecision)
            yPix = round(yPix, pixPrecision)
            point.AddPoint(xPix, yPix)
        geom_list.append([point, geom])

    for polygonTest in geom_list:
        
        if polygonTest[0].GetGeometryName() == 'POLYGON' or \
                        polygonTest[0].GetGeometryName() == 'LINESTRING' or \
                        polygonTest[0].GetGeometryName() == 'POINT':
            geom_pix_wkt_list.append([polygonTest[0].ExportToWkt(), polygonTest[1].ExportToWkt()])
        elif polygonTest[0].GetGeometryName() == 'MULTIPOLYGON' or \
                polygonTest[0].GetGeometryName() == 'MULTILINESTRING':
            for (pix,geo) in geom_list:
                geom_pix_wkt_list.append([pix.ExportToWkt(),geo.ExportToWkt()])

    return geom_pix_wkt_list


def convert_wgs84geojson_to_pixgeojson(wgs84geojson, inputraster, image_id=[], pixelgeojson=True,pixelgeojson_path='',
                                       breakMultiGeo=False, pixPrecision=2):
    
    dataSource = ogr.Open(wgs84geojson, 0)
    if dataSource is None:
        print '='*50
        print 'GeoJson {} has no Coordinates.'.format(wgs84geojson)
        print '='*50
        return
    layer = dataSource.GetLayer()
    #print(layer.GetFeatureCount())
    building_id = 0
    # check if geoJsonisEmpty
    feautureList = []
    if not image_id:
        image_id = inputraster.split('/')[-1].replace(".tif", "")

    if layer.GetFeatureCount() > 0:
        
        if len(inputraster)>0:
            if os.path.isfile(inputraster):
                srcRaster = gdal.Open(inputraster)
                targetSR = osr.SpatialReference()
                targetSR.ImportFromWkt(srcRaster.GetProjectionRef())
                geomTransform = srcRaster.GetGeoTransform()

                featureId = 0
                for feature in layer:
                    
                    geom = feature.GetGeometryRef()
                    road_id = feature.GetField('road_id')
                    featureName = 'roads'
                    if len(inputraster)>0:
                        ## Calculate 3 band
                        geom_wkt_list = geoWKTToPixelWKT(geom, inputraster, targetSR, geomTransform,breakMultiGeo,
                                                             pixPrecision=pixPrecision) 

                        for geom_wkt in geom_wkt_list:
                            featureId += 1
                            feautureList.append({'ImageId': image_id,
                                                 'RoadId': road_id,
                                                 'lineGeo': ogr.CreateGeometryFromWkt(geom_wkt[1]),
                                                 'linePix': ogr.CreateGeometryFromWkt(geom_wkt[0]),
                                                 'featureName': featureName,
                                                 'featureIdNum': featureId
                                                 })
                    else:
                        featureId += 1
                        feautureList.append({'ImageId': image_id,
                                             'RoadId': road_id,
                                             'lineGeo': ogr.CreateGeometryFromWkt(geom.ExportToWkt()),
                                             'linePix': ogr.CreateGeometryFromWkt('LINESTRING EMPTY'),
                                             'featureName' : featureName,
                                             'featureIdNum': featureId
                                             })
            else:
                #print("no File exists")
                pass
        if pixelgeojson:
            exporttogeojson(os.path.join(pixelgeojson_path,image_id+'.geojson'), buildinglist=feautureList)

    return feautureList

def exporttogeojson(geojsonfilename, buildinglist):
    #
    # geojsonname should end with .geojson
    # building list should be list of dictionaries
    # list of Dictionaries {'ImageId': image_id, 'RoadID': road_id, 'linePix': poly,
    #                       'lineGeo': poly}
    # image_id is a string,
    # BuildingId is an integer,
    # poly is a ogr.Geometry Polygon
    #
    # returns geojsonfilename

#     print geojsonfilename
    driver = ogr.GetDriverByName('geojson')
    if os.path.exists(geojsonfilename):
        driver.DeleteDataSource(geojsonfilename)
    datasource = driver.CreateDataSource(geojsonfilename)
    layer = datasource.CreateLayer('roads', geom_type=ogr.wkbLineString)
    field_name = ogr.FieldDefn("ImageId", ogr.OFTString)
    field_name.SetWidth(75)
    layer.CreateField(field_name)
    layer.CreateField(ogr.FieldDefn("RoadId", ogr.OFTInteger))

    # loop through buildings
    for building in buildinglist:
        # create feature
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("ImageId", building['ImageId'])
        feature.SetField("RoadId", building['RoadId'])
        feature.SetGeometry(building['linePix'])

        # Create the feature in the layer (geojson)
        layer.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()

    datasource.Destroy()

    return geojsonfilename

def ConvertTo8BitImage(srcFileName,outFileDir,outputFormat='GTiff'):
    
    outputPixType='Byte'
    srcRaster = gdal.Open(srcFileName)
    outputRaster = os.path.join(outFileDir, srcFileName.split('/')[-1])
    xmlFileName = outputRaster.replace('.tif','.xml')
    
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat, '-co', '"PHOTOMETRIC=rgb"']
    scaleList = []
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band=srcRaster.GetRasterBand(bandId)
        min = band.GetMinimum()
        max = band.GetMaximum()

        # if not exist minimum and maximum values
        if min is None or max is None:
            (min, max) = band.ComputeRasterMinMax(1)
        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(max))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(srcFileName)

    if outputFormat == 'JPEG':
        outputRaster = xmlFileName.replace('.xml', '.jpg')
    else:
        outputRaster = xmlFileName.replace('.xml', '.tif')

    outputRaster = outputRaster.replace('_img', '_8bit_img')
    
    cmd.append(outputRaster)
    print(' '.join(cmd))
    subprocess.call(cmd)
    
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def ConvertToRoadSegmentation(tif_file,geojson_file,out_file,isInstance=False):
    
    #Read Dataset from geo json file
    dataset = ogr.Open(geojson_file)
    if not dataset:
        print 'No Dataset'
        return -1
    layer = dataset.GetLayerByIndex(0)
    if not layer:
        print 'No Layer'
        return -1
    
    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(tif_file, gdal.GA_ReadOnly)

    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_file, ncol, nrow, 1, gdal.GDT_Byte)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)

    if isInstance:
        b.Fill(0)
        # Rasterize the shapefile layer to our new dataset
        status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                     [1],  # output to our new dataset's first band
                                     layer,  # rasterize this layer
                                     None, None,  # don't worry about transformations since we're in same projection
                                     [0],  # burn value 0
                                     ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                      'ATTRIBUTE=road_type']  # put raster values according to the 'id' field values
                                     )
    else:
        b.Fill(0)
        # Rasterize the shapefile layer to our new dataset
        status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                     [1],  # output to our new dataset's first band
                                     layer,  # rasterize this layer
                                     None, None,  # don't worry about transformations since we're in same projection
                                     [255]  # burn value 0
                                     )

    # Close dataset
    out_raster_ds = None
    
    return status

def CreateEmptyTif(tif_file,out_file):
    
    #Read Dataset from geo json file
    layer = None
    
    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(tif_file, gdal.GA_ReadOnly)

    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_file, ncol, nrow, 1, gdal.GDT_Byte)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)

    b.Fill(0)
    # Rasterize the shapefile layer to our new dataset
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 #None,  # rasterize this layer
                                 None, None,  # don't worry about transformations since we're in same projection
                                 [255]  # burn value 0
                                 )

    # Close dataset
    out_raster_ds = None
    
    return status