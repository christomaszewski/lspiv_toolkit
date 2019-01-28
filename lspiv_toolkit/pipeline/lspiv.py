import cv2
import numpy as np
import os
import time
import glob
import time

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ShiTomasiDetector
from cv_toolkit.detect.adapters import GridDetector

from cv_toolkit.track.flow import LKOpticalFlowTracker

from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform
from cv_toolkit.transform.common import IdentityTransform

import field_toolkit.approx as field_approx

from primitives import Track, Grid

from ..filtering.tracks import TrackDB
from ..filtering.measurements import MeasurementDB

class SlimPipeline(object):
	""" A slim version of the lspiv pipeline that just processes the entire
		dataset in the background without displaying any images to the screen

	"""

	def __init__(self, config=None):
		if (config is not None):
			self.load(config)
		else:
			self._config = None


	def load(self, config):
		self._config = config

		# Load Dataset
		self._data = Dataset.from_file(config.datasetFile)
		#self._imgWidth, self._imgHeight = self._data.imgSize

		# Setup transformation objects - not needed anymore
		#self._undistortTransform = UndistortionTransform(self._data.camera)
		#self._pxTransform = PixelCoordinateTransform(self._data.imgSize)

		# Initialize grid object for measurement filtering
		self._detectionGrid = Grid(*self._data.imgSize, *config.detectionGridDim)

		# Instantiate detector
		self._detector = ShiTomasiDetector(**config.getFeatureDetectionParams())

		# Setup Track and Measurement databases
		self._tDB = TrackDB(**config.getTrackFilteringParams())

		# Setup Grid Detector
		self._gd = GridDetector.from_grid(self._detector, self._detectionGrid, config.maxFeatures, config.borderBuffer)

		# Setup LKTracker with default params
		self._lk = LKOpticalFlowTracker(**config.getLKFlowParams())

	def initialize(self):
		""" Initialize new output folder and prepare pipeline for execution

		"""

		# Initialize output folders
		self._outputDir = f"{self._config.outputDir}/{self._data.name}"
		if not os.path.exists(self._outputDir):
			os.makedirs(self._outputDir)

			# If top level output dir was just created, save dataset and cam files
			self._data.camera.save(f"{self._outputDir}/camera.yaml")
			self._data.save(f"{self._outputDir}/dataset.yaml")

		self._runDir =  f"{self._outputDir}/lspiv_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
		if not os.path.exists(self._runDir):
			os.makedirs(self._runDir)

		self._trackDir =  f"{self._runDir}/tracks"
		if not os.path.exists(self._trackDir):
			os.makedirs(self._trackDir)

		self._rawTrackDir =  f"{self._trackDir}/raw"
		if not os.path.exists(self._rawTrackDir):
			os.makedirs(self._rawTrackDir)

		self._prunedTrackDir = f"{self._trackDir}/pruned"
		if not os.path.exists(self._prunedTrackDir):
			os.makedirs(self._prunedTrackDir)

		# Save pruned historical tracks to file
		self._tDB.savePrunedTracks(self._prunedTrackDir)

		# Save pipeline config file to run dir
		self._config.save(f"{self._runDir}/pipeline_config.yaml")

		# Detect Initial Features
		img, timestamp = self._data.read()
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		points = self._gd.detect(grayImg, self._data.mask)
		tracks = [Track.from_point(p, timestamp) for p in points]

		# Instantiate tracks 
		self._tDB.addNewTracks(tracks)

		# Initialize LK Tracker with first image
		self._lk.loadImage(grayImg)

		# Initialize timestamps for pipeline control
		self._lastDetectionTime = timestamp
		self._lastTimestamp = timestamp

	def run(self):
		startTime = time.time()
		while(self._data.more()):
			# Load next image
			img, timestamp = self._data.read()
			progress = self._data.progress
			
			timeElapsed = time.time() - startTime
			executionRate = progress / timeElapsed
			eta = (100.0 - progress) / executionRate
			print(f"Dataset {progress:.2f}% Processed, Current Timestamp: {timestamp:.3f}, Estimated Time Left: {eta:.3f}s")

			# Convert image to grayscale for detection and tracking
			grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Get current track end points
			endPoints = np.asarray(self._tDB.getActiveEndpoints())
			# Attempt to track end points using LK optical flow
			newPoints = self._lk.trackPoints(endPoints, grayImg)
		
			# Update track end points with results from LK tracker
			self._tDB.updateActiveTracks(newPoints, timestamp)

			# If active tracks are low or it is time to run a detection, do so
			if (self._tDB.getNumActiveTracks() < self._config.numDesiredTracks or timestamp - self._lastDetectionTime > self._config.detectionInterval):
				# Mask active track end points
				searchMask = np.copy(self._data.mask)
				endPoints = self._tDB.getActiveEndpoints()

				for point in endPoints:
					cv2.circle(searchMask, tuple(point), 5, 0, -1)

				detections = self._gd.detect(grayImg, searchMask)
				
				self._tDB.addNewTracks([Track.from_point(p, timestamp) for p in detections])
				self._lastDetectionTime = timestamp
				del searchMask, detections

			self._lastTimestamp = timestamp

			del grayImg, img

		self._tDB.terminateActiveTracks()

		totalTime = time.time() -  startTime
		print(f"Pipeline run complete in {totalTime} seconds")

	def saveTracks(self, timestamp=None):
		if timestamp is None:
			timestamp = self._lastTimestamp

		tracks = self._tDB.getHistoricalTracks()
		
		for t in tracks:
			t.save(f"{self._rawTrackDir}/track_{t.id}.json")

	@property
	def runDir(self):
		return self._runDir

	@property
	def outputDir(self):
		return self._outputDir