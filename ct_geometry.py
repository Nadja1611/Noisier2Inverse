# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import pathlib

import numpy as np
import json

from LION.utils.parameter import LIONParameter


class Geometry(LIONParameter):
    """
    Class holding a CT geometry
    """

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", None)

        self.image_size = np.array(kwargs.get("image_size", None))
        self.image_shape = np.array(kwargs.get("image_shape", None))
        if self.image_shape.all() and self.image_size.all():
            self.voxel_size = self.image_size / self.image_shape

        self.detector_shape = np.array(kwargs.get("detector_shape", None))
        self.detector_size = np.array(kwargs.get("detector_size", None))
        if self.detector_size.all() and self.detector_shape.all():
            self.pixel_size = self.detector_size / self.detector_shape

        self.dso = np.array(kwargs.get("dso", None))
        self.dsd = np.array(kwargs.get("dsd", None))

        self.angles = np.array(kwargs.get("angles", None))

    @staticmethod
    def default_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )

    @staticmethod
    def sparse_view_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 50, endpoint=False),
        )

    @staticmethod
    def sparse_angle_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi / 6, 60, endpoint=False),
        )

    @staticmethod
    def parallel_default_parameters(image_shape=None, number_of_angles=360, translate=False, angle_vector = np.array([23]), folds = 4):
        if image_shape is None:
            image_shape = [1, 512, 512]
        if translate == False:    
            return Geometry(
                image_shape=image_shape,
                image_size=image_shape,
                detector_shape=image_shape[0:2],
                detector_size=image_shape[0:2],
                dso=image_shape[1] * 2,
                dsd=image_shape[1] * 4,
                mode="parallel",
                angles=np.linspace(0, np.pi, number_of_angles, endpoint=False))
        
        if translate == "N2I":    
            if angle_vector is not None and angle_vector[0] is not None:
                return Geometry(
                    image_shape=image_shape,
                    image_size=image_shape,
                    detector_shape=image_shape[0:2],
                    detector_size=image_shape[0:2],
                    dso=image_shape[1] * 2,
                    dsd=image_shape[1] * 4,
                    mode="parallel",
                    angles=np.linspace(0, np.pi, int(folds)*number_of_angles)[int(angle_vector[0])::int(folds)]
                )
            else:
                # Handle the case where angle_vector[0] is None
                # You might want to raise an exception or handle it in an appropriate way
                pass

        else:
            return Geometry(

               image_shape=image_shape,
               image_size=image_shape,
               detector_shape=image_shape[0:2],
               detector_size=image_shape[0:2],
               dso=image_shape[1] * 2,
               dsd=image_shape[1] * 4,
               mode="parallel",
               angles=np.linspace(np.pi/(number_of_angles*2), np.pi+np.pi/(number_of_angles*2), number_of_angles, endpoint=False),
            
        )


    def parallel_default_parameters_n2i(image_shape=None, angles_n2i=np.linspace(0,np.pi, 64, endpoint=False)):
        if image_shape is None:
            image_shape = [1, 512, 512]
            
        return Geometry(
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=angles_n2i,
        )

    @staticmethod
    def parallel_sparse_view_parameters(image_shape=None):
        if image_shape is None:
            image_shape = [1, 512, 512]
        return Geometry(
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=np.linspace(0, 2 * np.pi, 50, endpoint=False),
        )

    staticmethod

    def parallel_sparse_angle_parameters(image_shape=None):
        if image_shape is None:
            image_shape = [1, 512, 512]
        return Geometry(
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=np.linspace(0, 2 * np.pi / 6, 60, endpoint=False),
        )

    def default_geo(self):
        self.__init__(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )

    def __str__(self):
        string = []
        string.append("CT Geometry")
        string.append("CT type/mode = " + self.mode)
        string.append("-----")
        string.append(
            "Distance from source to detector (DSD) = " + str(self.dsd) + " mm"
        )
        string.append("Distance from source to origin (DSO)= " + str(self.dso) + " mm")
        string.append("-----")
        string.append("Detector")
        string.append("Detector shape = " + str(self.detector_shape))
        string.append("Detector size = " + str(self.detector_size) + " mm")
        string.append("-----")
        string.append("Image")
        string.append("Image shape = " + str(self.image_shape))
        string.append("Image size = " + str(self.image_size) + " mm")
        string.append("Number of angles = " + str(self.angles.shape))
        return "\n".join(string)
