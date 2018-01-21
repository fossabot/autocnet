import argparse
import os
import sys

import numpy as np
import pandas as pd

from autocnet.matcher.cuda_extractor import extract_features
from autocnet.utils.utils import tile
from autocnet.io.keypoints import to_hdf
from autocnet.camera.csm_camera import create_camera

from plio.io.io_gdal import GeoDataset

import autocnet
funcs = {'vlfeat':autocnet.matcher.cpu_extractor.extract_features}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument('-t', '--threshold', help='The threshold difference between DN values')
    parser.add_argument('-n', '--nfeatures', help='The number of features to extract. Default is max_image_dimension / 1.25', type=float)
    parser.add_argument('-m', '--maxsize',type=float, default=6e7, help='The maximum number of pixels before tiling is used to extract keypoints.  Default: 6e7')
    parser.add_argument('-e', '--extractor', default='vlfeat', choices=['cuda', 'vlfeat'], help='The extractor to use to get keypoints.')
    parser.add_argument('-c', '--camera', action='store_false', help='Whether or not to compute keypoints coordinates in body fixed as well as image space.')
    parser.add_argument('-o', '--outdir', type=str, help='The output directory')
    return vars(parser.parse_args())

def extract_features(self, array, xystart=[], *args, **kwargs):
    keypoints, descriptors = Node._extract_features(array, *args, **kwargs)

    count = len(self.keypoints)
    # If this is a tile, push the keypoints to the correct start xy
    if xystart:
        keypoints['x'] += xystart[0]
        keypoints['y'] += xystart[1]

    self.keypoints = pd.concat((self.keypoints, keypoints))
    descriptor_mask = self.keypoints[count:].duplicated()
    number_new = len    (descriptor_mask) - descriptor_mask.sum()
    # Removed duplicated and re-index the merged keypoints
    self.keypoints.drop_duplicates(inplace=True)
    self.keypoints.reset_index(inplace=True, drop=True)

    if self.descriptors is not None:
        self.descriptors = np.concatenate((self.descriptors, descriptors[~descriptor_mask]))
    else:
        self.descriptors = descriptors
    #self.descriptors = descriptors
    assert count + number_new == len(self.descriptors)

def extract(ds, extractor, maxsize):
    array_size = ds.raster_size[0] * ds.raster_size[1]
    if array_size > maxsize:
        slices = tile(array_size, tilesize=12000, overlap=250)
        for s in slices:
            xystart = [s[0], s[1]]
            array = ds.read_array(pixels=s)
            #extract_features(array, xystart, *args, **kwargs)
    else:
        array = ds.read_array()
        extractor_params = {'compute_descriptor': True,
                            'float_descriptors': True,
                            'edge_thresh':2.8,
                            'peak_thresh': 0.0001,
                            'verbose': False}
        keypoints, descriptors = funcs[extractor](array, extractor_method='vlfeat', extractor_parameters=extractor_params)
    return keypoints, descriptors


def create_output_path(ds, outdir):
    image_name = os.path.basename(ds.file_name)
    image_path = os.path.dirname(ds.file_name)

    if outdir is None:
        outh5 = os.path.join(image_path, image_name + '_kps.h5')
    else:
        outh5 = os.path.join(outdir, image_name + '_kps.h5')

    return outh5

if __name__ == '__main__':
    # Setup the metadata obj that will be written to the db
    metadata = {}

    # Parse args and grab the file handle to the image
    kwargs = parse_args()
    input_file = kwargs.pop('input_file', None)
    ds = GeoDataset(input_file)

    # Extract the correspondences
    extractor = kwargs.pop('extractor')
    maxsize = kwargs.pop('maxsize')
    keypoints, descriptors = extract(ds, extractor, maxsize)

    # Create a camera model for the image
    camera = kwargs.pop('camera')
    if camera:
        camera = create_camera(ds)

        # Project the sift keypoints to the ground
        def func(row, args):
            camera = args[0]
            gnd = getattr(camera, 'imageToGround')(row[1], row[0], 0)
            return gnd

        feats = keypoints[['x', 'y']].values
        gnd = np.apply_along_axis(func, 1, feats, args=(camera, ))
        gnd = pd.DataFrame(gnd, columns=['xm', 'ym', 'zm'], index=keypoints.index)
        keypoints = pd.concat([keypoints, gnd], axis=1)

        # Write metadata about the keypoints

    # Write the correspondences to disk
    outdir = kwargs.pop('outdir')
    outpath = create_output_path(ds, outdir)
    to_hdf(keypoints, descriptors, outpath)

    # Write to the database indicating completion

    metadata = {'image_name':image_name,
                'image_path':image_path,
                'nkeypoints':len(keypoints),
                'keypoint_path':outh5}




    # Then send back the necessary metadata.
