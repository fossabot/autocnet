import cv2
import pandas as pd

import numpy as np
from skimage.feature import match_template
from scipy.misc import imresize

FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE,
                                trees=3)


def pattern_match(template, image, upsampling=10,
                  func=match_template):
    """
    Call an arbitrary pattern matcher

    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image

    image : ndarray
            The image or sub-image to be searched

    upsampling : int
                 The multiplier to upsample the template and image.

    func : object
           The function to be used to perform the template based matching

    Returns
    -------

    x : float
        The x offset

    y : float
        The y offset

    strength : float
               The strength of the correlation in the range [-1, 1].
    """

    if upsampling < 1:
        raise ValueError

    u_template = imresize(template, (template.shape[0] * upsampling,
                                   template.shape[1] * upsampling),
                        interp='bicubic')

    u_image = imresize(image, (image.shape[0] * upsampling,
                             image.shape[1] * upsampling),
                     interp='bicubic')

    # Find the the upper left origin of the template in the image
    match = func(u_image, u_template)
    y, x = np.unravel_index(np.argmax(match), match.shape)

    # Resample the match back to the native image resolution
    x /= upsampling
    y /= upsampling

    # Offset from the UL origin to the image center
    x += (template.shape[1] / 2)
    y += (template.shape[0] / 2)

    # Compute the offset to adjust the image match point location
    ideal_y = image.shape[0] / 2
    ideal_x = image.shape[1] / 2

    x = ideal_x - x
    y = ideal_y - y

    # Find the maximum correlation
    strength = np.max(match)

    return x, y, strength


class FlannMatcher(object):
    """
    A wrapper to the OpenCV Flann based matcher class that adds
    metadata tracking attributes and methods.  This takes arbitrary
    descriptors and so should be available for use with any
    descriptor data stored as an ndarray.

    Attributes
    ----------
    image_indices : dict
                    with key equal to the train image index (returned by the DMatch object),
                    e.g. an integer array index
                    and value equal to the image identifier, e.g. the name

    image_index_counter : int
                          The current number of images loaded into the matcher
    """

    def __init__(self, flann_parameters=DEFAULT_FLANN_PARAMETERS):
        self._flann_matcher = cv2.FlannBasedMatcher(flann_parameters, {})
        self.image_indices = {}
        self.image_index_counter = 0

    def add(self, descriptor, key):
        """
        Add a set of descriptors to the matcher and add the image
        index key to the image_indices attribute

        Parameters
        ----------
        descriptor : ndarray
                     The descriptor to be added

        key : hashable
              The identifier for this image, e.g. the image name
        """
        self._flann_matcher.add([descriptor])
        self.image_indices[self.image_index_counter] = key
        self.image_index_counter += 1

    def train(self):
        """
        Using the descriptors, generate the KDTree
        """
        self._flann_matcher.train()

    def query(self, descriptor, query_image, k=3):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        query_image : hashable
                      Key of the query image

        k : int
            The number of nearest neighbors to search for

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        """

        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for m in matches:
            for i in m:
                # This checks for self neighbor and never allows them into the graph
                if self.image_indices[i.imgIdx] == query_image:
                    continue

                # Ensure ordering in the source / destination
                if query_image < self.image_indices[i.imgIdx]:
                    matched.append((query_image,
                                    i.queryIdx,
                                    self.image_indices[i.imgIdx],
                                    i.trainIdx,
                                    i.distance))
                else:
                    matched.append((self.image_indices[i.imgIdx],
                                    i.trainIdx,
                                    query_image,
                                    i.queryIdx,
                                    i.distance))
        return pd.DataFrame(matched, columns=['source_image', 'source_idx',
                                              'destination_image', 'destination_idx',
                                              'distance'])
