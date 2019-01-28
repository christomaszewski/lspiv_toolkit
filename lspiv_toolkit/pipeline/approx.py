import glob
import time
import os

from primitives.track import Track
from primitives.grid import Grid

from cv_toolkit.cams import FisheyeCamera
from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform

from ..filtering.measurements import MeasurementDB

import field_toolkit.approx as field_approx

class ApproximationPipeline(object):

	def __init__(self, config=None):
		if config is not None:
			self.load(config)
		else:
			self._config = None

	def load(self, config):
		self._config = config

		# Load camera from file
		self._camera = FisheyeCamera.from_file(config.camFile)

		# Set input and track directories
		self._inputDir = config.inputDir		
		self._trackDir =  f"{self._inputDir}/tracks"

		# Initialize grid for measurement filtering
		self._measurementGrid = Grid(*self._camera.imgSize, *config.measurementGridDim)

		# Initialize measurement filtering database
		self._mDB = MeasurementDB(self._measurementGrid, **config.getFilteringParams())

		# Initialize transformations
		self._unTrans = UndistortionTransform(self._camera)
		self._pxTrans = PixelCoordinateTransform(self._camera.imgSize)

		# Initialize approximation object
		if config.approximationMethod == 'simple':
			self._gp = field_approx.gp.GPApproximator()
		elif config.approximationMethod == 'coregionalized':
			self._gp = field_approx.gp.CoregionalizedGPApproximator()
		elif config.approximationMethod == 'sparse':
			self._gp = field_approx.gp.SparseGPApproximator()
		elif config.approximationMethod == 'integral':
			self._gp = field_approx.gp.IntegralGPApproximator()
		else:
			print("Error: Unknown approximation method")
			exit()

	def initialize(self):
		# Initialize output folders
		self._runDir =  f"{self._inputDir}/approx_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
		if not os.path.exists(self._runDir):
			os.makedirs(self._runDir)

		self._measurementDir =  f"{self._runDir}/measurements"
		if not os.path.exists(self._measurementDir):
			os.makedirs(self._measurementDir)

		# Save config file
		self._config.save(f"{self._runDir}/approx_config.yaml")

		# Clear any previous measurements from database or gp approx
		self._mDB.clearMeasurements()
		self._gp.clearMeasurements()

	def run(self):
		startTime = time.time()

		# Load training tracks
		trackFiles = []
		for subset in self._config.trainingSets:
			trackFiles.extend(glob.glob(f"{self._trackDir}/{subset}/track_*.json"))

		print(f"Loading {len(trackFiles)} tracks")

		tracks = Track.from_file_list(trackFiles)

		transformedTracks = self._pxTrans.transformTracks(self._unTrans.transformTracks(tracks))

		for t in transformedTracks:
			self._mDB.addMeasurements(t.measureVelocity(**self._config.getMeasurementParams(), **self._config.measurementMethodParams))

		self._trainingMeasurements = self._mDB.getMeasurements(self._config.measurementsPerCell)

		if len(self._trainingMeasurements) > 0:
			self._gp.clearMeasurements()
			self._gp.addMeasurements(self._trainingMeasurements)
			self._fieldApprox = self._gp.approximate()

		totalTime = time.time() - startTime
		print(f"Approximation complete in {totalTime} seconds")

	def saveApproximation(self):
		self._fieldApprox.save(f"{self._runDir}/approx.field")

	def saveMeasurements(self):
		# Saves tracks used for approximation
		for i, m in enumerate(self._trainingMeasurements):
			m.save(f"{self._measurementDir}/measurement_{i}.json")

		# Todo save measurement database images/data

	@property
	def runDir(self):
		return self._runDir