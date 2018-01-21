import os
import json
import datetime
import requests

import numpy as np

import usgscam as cam
from cycsm.isd import Isd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# Utility Func for working with PVL
def find_in_dict(obj, key):
    """
    Recursively find an entry in a dictionary

    Parameters
    ----------
    obj : dict
          The dictionary to search
    key : str
          The key to find in the dictionary

    Returns
    -------
    item : obj
           The value from the dictionary
    """
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = find_in_dict(v, key)
            if item is not None:
                return item

def data_from_cube(header):
    data = {}
    data['START_TIME'] = find_in_dict(header, 'StartTime')
    data['SPACECRAFT_NAME'] = find_in_dict(header, 'SpacecraftName')
    data['INSTRUMENT_NAME'] = find_in_dict(header, 'InstrumentId')
    data['SAMPLING_FACTOR'] = find_in_dict(header, 'SpatialSumming')
    data['SAMPLE_FIRST_PIXEL'] = find_in_dict(header, 'SampleFirstPixel')
    data['IMAGE'] = {}
    data['IMAGE']['LINES'] = find_in_dict(header, 'Lines')
    data['TARGET_NAME'] = find_in_dict(header, 'TargetName')
    data['LINE_EXPOSURE_DURATION'] = find_in_dict(header, 'LineExposureDuration')
    data['SPACECRAFT_CLOCK_START_COUNT'] = find_in_dict(header, 'SpacecraftClockCount')
    return data

def create_camera(obj, url='http://localhost:5000/api/1.0/missions/mars_reconnaissance_oribter/ctx/csm_isd'):
    data = json.dumps(data_from_cube(obj.metadata), cls=NumpyEncoder)
    r = requests.post(url, data=data)


    # Get the ISD back and instantiate a local ISD for the image
    isd = r.json()['data']['isd']
    i = Isd.loads(isd)

    # Create the plugin and camera as usual
    plugin = cam.genericls.GenericLsPlugin()
    return plugin.from_isd(i, plugin.modelname(0))
