import cv2
import numpy as np
import os
import time
import imageio
import glob
import matplotlib.pyplot as plt

from context import lspiv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ShiTomasiDetector
from cv_toolkit.detect.adapters import GridDetector

from cv_toolkit.track.flow import LKOpticalFlowTracker

from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform
from cv_toolkit.transform.common import IdentityTransform

import field_toolkit.approx as field_approx

from primitives.track import Track
from primitives.grid import Grid

from lspiv_toolkit.filtering.tracks import TrackDB
from lspiv_toolkit.filtering.measurements import MeasurementDB

from viz_toolkit.view import ImageView, OverlayView, FieldOverlayView

from lspiv_toolkit.pipeline import BasicPipeline, SlimPipeline

def generateGifs():
	# Gif generation
	measurementGlob = sorted(glob.glob(f"{runDir}measurements_*.png"), key=os.path.getmtime)
	sourceGlob = sorted(glob.glob(f"{runDir}source_*.png"), key=os.path.getmtime)
	approxGlob = sorted(glob.glob(f"{runDir}undistorted_*.png"), key=os.path.getmtime)

	images = []

	for file in measurementGlob:
		images.append(imageio.imread(file))

	imageio.mimsave(f"{runDir}measurements.gif", images, duration=1, loop=2)

	images = []

	for file in sourceGlob:
		images.append(imageio.imread(file))

	imageio.mimsave(f"{runDir}source.gif", images, duration=1, loop=2)

	images = []

	for file in approxGlob:
		images.append(imageio.imread(file))

	imageio.mimsave(f"{runDir}undistorted.gif", images, duration=1, loop=2)


if __name__ == '__main__':
	datasetName = 'llobregat_full'
	datasetFilename = '../../../datasets/' + datasetName + '.yaml'

	outputDir = '../../../output/'

	pipeline = SlimPipeline(datasetFilename, outputDir)
	pipeline.initialize()
	pipeline.run()
	pipeline.saveTracks()
	pipeline.saveApproximation()

	plt.close('all')
	
	# datasetName = 'llobregat_15min'
	# datasetFilename = '../../../datasets/' + datasetName + '.yaml'

	# pipeline.load(datasetFilename)
	# pipeline.initialize()
	# pipeline.loop()

	# plt.close('all')
	# datasetName = 'llobregat_2nd_15min'
	# datasetFilename = '../../../datasets/' + datasetName + '.yaml'

	# pipeline.load(datasetFilename)
	# pipeline.initialize()
	# pipeline.loop()

	cv2.destroyAllWindows()