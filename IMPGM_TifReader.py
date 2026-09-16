from osgeo import gdal, osr
import numpy as np
import os

gdal.UseExceptions()

class Tif_Read_and_Write(object):
    """Read and write georeferenced TIF images with GDAL."""
    def __init__(self):
        super(Tif_Read_and_Write, self).__init__()

    def Tif_Read(self, input_data_path):
        dataset = gdal.Open(input_data_path)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_proj = dataset.GetProjection()
        im_Geotrans = dataset.GetGeoTransform()
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
        del dataset
        return im_data, im_proj, im_Geotrans


    def Numpy_to_Tif(self, array_data, output_path, top_left_lon, top_left_lat, pixel_width, pixel_height, epsg_code=None, prj_info=None, nodata_value=np.nan):

        if len(array_data.shape) == 3:  # check input dimensions: single-band or multi-band
            im_bands, rows, cols = array_data.shape
        else:
            im_bands, (rows, cols) = 1, array_data.shape
        geotransform = (top_left_lon, pixel_width, 0, top_left_lat, 0, pixel_height)     # note: pixel_height should be negative since row coordinates decrease as latitude increases
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(output_path, cols, rows, im_bands, gdal.GDT_Float32)
        output_dataset.SetGeoTransform(geotransform)
        output_srs = osr.SpatialReference()
        if prj_info is not None:
            output_srs.ImportFromWkt(prj_info)
        elif epsg_code is not None:
            output_srs.ImportFromEPSG(epsg_code)
        else:
            raise Exception("Either epsg_code or prj_content must be provided.")
        
        output_dataset.SetProjection(output_srs.ExportToWkt())
        if im_bands == 1:
            output_band = output_dataset.GetRasterBand(1)
            output_band.SetNoDataValue(nodata_value)
            output_band.WriteArray(array_data[0] if array_data.ndim == 3 else array_data)
        else:
            for i in range(im_bands):
                output_band = output_dataset.GetRasterBand(i + 1)
                output_band.SetNoDataValue(nodata_value)
                output_band.WriteArray(array_data[i])
        output_dataset.BuildOverviews("BILINEAR", [2, 4, 8, 16])
        del output_dataset
