import cv2
import numpy as np
import os
import time
import imageio
import glob
import dill

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ShiTomasiDetector
from cv_toolkit.detect.adapters import GridDetector

from cv_toolkit.track.flow import LKOpticalFlowTracker

from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform
from cv_toolkit.transform.common import IdentityTransform

import field_toolkit.approx as field_approx
import field_toolkit.viz as field_viz

from primitives.track import Track
from primitives.grid import Grid

from viz_toolkit.view import ImageView, OverlayView, FieldOverlayView

from .filtering.tracks import TrackDB
from .filtering.measurements import MeasurementDB

class BasicPipeline(object):

	def __init__(self, datasetFile, outputDir):
		self._datasetFilename = datasetFile
		self._outputDir = outputDir

		# Intervals (seconds)
		self._detectionInterval = 3
		self._approximationInterval = 100000
		self._inputImgSaveInterval = 5
		self._measurementImgSaveInterval = 5

		# Feature params
		self._desiredActiveTracks = 1000
		self._detectionLimit = 30001

	def load(self, datasetFile):
		self._datasetFilename = datasetFile


	def initialize(self):
		# Load dataset from file
		self._data = Dataset.from_file(self._datasetFilename)
		self._imgWidth, self._imgHeight = self._data.imgSize

		# Initialize output folders
		self._datasetOutputDir = self._outputDir + self._data.name
		if not os.path.exists(self._datasetOutputDir):
			os.makedirs(self._datasetOutputDir)

		self._runDir =  self._datasetOutputDir + '/' + time.strftime('%d_%m_%Y_%H_%M_%S/')
		if not os.path.exists(self._runDir):
			os.makedirs(self._runDir)

		# Initialize grid object for measurement filtering
		self._measurementGrid = Grid(self._imgWidth, self._imgHeight, 200, 100)
		self._detectionGrid = Grid(self._imgWidth, self._imgHeight, 200, 30)
		self._approxVizGrid = Grid(self._imgWidth, self._imgHeight, 40, 30)

		# Setup image display
		self._imgView = ImageView(windowName='Input')
		self._undistortedView = ImageView(windowName='Undistorted')
		self._measurementView = OverlayView(grid=self._measurementGrid)
		self._approxView = FieldOverlayView(grid=self._approxVizGrid)

		# Setup Track and Measurement databases
		self._tDB = TrackDB()
		self._mDB = MeasurementDB(self._measurementGrid, 5)

		# Instantiate detector
		self._detector = ShiTomasiDetector()

		# Setup Grid Detector
		self._gd = GridDetector.from_grid(self._detector, self._detectionGrid, self._detectionLimit, 50)

		# Detect Initial Features
		img, timestamp = self._data.read()
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		points = self._gd.detect(grayImg, self._data.mask)

		# Instantiate tracks 
		self._tDB.addNewTracks([Track.from_point(p, timestamp) for p in points])

		# Setup LKTracker with default params
		self._lk = LKOpticalFlowTracker(grayImg)

		# Setup GPR field approximator with default params
		self._gp = field_approx.gp.GPApproximator()

		# Setup transformation objects
		self._undistortTransform = UndistortionTransform(self._data.camera)
		self._pxTransform = PixelCoordinateTransform(self._data.imgSize)

		# Plot and save initial images
		undistortedImg = self._undistortTransform.transformImage(img)
		self._imgView.updateImage(img, timestamp)
		self._undistortedView.updateImage(undistortedImg, timestamp)
		measurementImg = self._pxTransform.transformImage(undistortedImg)
		self._measurementView.updateImage(measurementImg, timestamp)
		self._approxView.updateImage(measurementImg, timestamp)

		self._imgView.save(f"{self._runDir}source_{timestamp:.2f}.png")
		self._undistortedView.save(f"{self._runDir}undistorted_{timestamp:.2f}.png")
		self._measurementView.save(f"{self._runDir}measurements_{timestamp:.2f}.png")


		# Initialize timestamps for pipeline control
		self._lastDetectionTime = timestamp
		self._lastApproximated = timestamp
		self._lastSavedInput = timestamp
		self._lastSavedMeasurements = timestamp

		self._lastTimestamp = timestamp

	def loop(self):
		while(self._data.more()):
			# Load next image
			img, timestamp = self._data.read()

			# Convert image for detection, tracking, and plotting
			grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			undistortedImg = self._undistortTransform.transformImage(img)

			# Update image views
			# Disabled for now
			#self._imgView.updateImage(img, timestamp)
			#self._undistortedView.updateImage(undistortedImg, timestamp)

			# Get current track end points
			endPoints = np.asarray(self._tDB.getActiveEndpoints())
			# Attempt to track end points using LK optical flow
			newPoints = self._lk.trackPoints(endPoints, grayImg)
			# Update track end points with results from LK tracker
			self._tDB.updateActiveTracks(newPoints, timestamp)

			# Pull tracks from Track DB
			activeTracks = self._tDB.getActiveTracks()
			#historicalTracks = self._tDB.getHistoricalTracks()
			#unwarpedHistorical = [self._undistortTransform.transformTrack(t) for t in historicalTracks]

			# Plot Track overlays
			# Disabled for now
			#self._imgView.plotTracksColoredByScore(activeTracks)
			#self._imgView.plot()
			#self._undistortedView.plotTracksColoredByScore(unwarpedHistorical)
			#self._undistortedView.plot()

			# If active tracks are low or it is time to run a detection, do so
			if (len(activeTracks) < self._desiredActiveTracks or timestamp - self._lastDetectionTime > self._detectionInterval):
				# Mask active track end points
				searchMask = np.copy(self._data.mask)
				for point in self._tDB.getActiveEndpoints():
					cv2.circle(searchMask, tuple(point), 5, 0, -1)

				detections = self._gd.detect(grayImg, searchMask)
				self._tDB.addNewTracks([Track.from_point(p, timestamp) for p in detections])
				self._lastDetectionTime = timestamp



			# If its time to save the input images, do so
			if (timestamp - self._lastSavedInput > self._inputImgSaveInterval):
				# Update image views
				self._imgView.updateImage(img, timestamp)
				self._imgView.plotTracksColoredByScore(activeTracks)
				self._imgView.plot()
				self._undistortedView.updateImage(undistortedImg, timestamp)
				self._imgView.save(f"{self._runDir}source_{timestamp:.2f}.png")
				self._undistortedView.save(f"{self._runDir}undistorted_{timestamp:.2f}.png")

				self._lastSavedInput = timestamp

			# If its time to save measurement image, generate it and save
			if (timestamp - self._lastSavedMeasurements > self._measurementImgSaveInterval):
				# Update Measurement DB
				self._mDB.clearMeasurements()
				historicalTracks = self._tDB.getHistoricalTracks()
				unwarpedHistorical = [self._undistortTransform.transformTrack(t) for t in historicalTracks]
				tracksForMeasurement = [self._pxTransform.transformTrack(t) for t in unwarpedHistorical]
				for t in tracksForMeasurement:
					self._mDB.addMeasurements(t.measureVelocity(minDist=0.0, scoring='composite'))

				measurementImg = self._pxTransform.transformImage(undistortedImg)
				self._measurementView.updateImage(measurementImg, timestamp)
				self._measurementView.hist2d(self._mDB.getMeasurements())
				self._measurementView.plot()

				self._measurementView.save(f"{self._runDir}measurements_{timestamp:.2f}.png")

				self._lastSavedMeasurements = timestamp


			# If it is time to run a field approximation, do so
			if (timestamp - self._lastApproximated > self._approximationInterval):
				print("running gpr")
				measurements = self._mDB.getMeasurements(measurementsPerCell=1)
				if len(measurements) > 0:
					self._gp.clearMeasurements()
					self._gp.addMeasurements(measurements)
					approxField = self._gp.approximate()

					self._approxView.updateImage(measurementImg, timestamp)

					self._approxView.drawField(approxField)

					self._approxView.save(f"{self._runDir}approx_{timestamp:.2f}.png")

				self._lastApproximated = timestamp

			if cv2.waitKey(1) & 0xFF == 27:
				break

			self._lastTimestamp = timestamp

			del grayImg, img

	def saveApproximation(self, timestamp=None):
		if timestamp is None:
			timestamp = self._lastTimestamp

		print("running gpr")
		measurements = self._mDB.getMeasurements(measurementsPerCell=1)
		if len(measurements) > 0:
			self._gp.clearMeasurements()
			self._gp.addMeasurements(measurements)
			approxField = self._gp.approximate()
		else:
			print("no measurements")
			return

		print("saving approx field to file")

		approxFieldFile = self._runDir + f"approx_{timestamp:.2f}.field"
		with open(approxFieldFile, mode='wb') as f:
			dill.dump(approxField, f)



	def saveTracks(self, timestamp=None):
		if timestamp is None:
			timestamp = self._lastTimestamp

		self._trackDir =  self._runDir + f"tracks_{timestamp:.2f}/"
		if not os.path.exists(self._trackDir):
			os.makedirs(self._trackDir)

		tracks = self._tDB.getHistoricalTracks()

		for i, t in enumerate(tracks):
			t.save(self._trackDir + f"track_{i}.yaml")


class SlimPipeline(object):
	""" A slim version of the lspiv pipeline that just processes the entire
		dataset in the background without displaying any images to the screen

	"""

	def __init__(self, datasetFile, outputDir):
		self._outputDir = outputDir

		# Intervals (seconds)
		self._detectionInterval = 3

		# Feature params
		self._desiredActiveTracks = 1000
		self._detectionLimit = 30001

		# Instantiate detector
		self._detector = ShiTomasiDetector()

		# Load dataset
		self.load(datasetFile)


	def load(self, datasetFile):
		self._datasetFilename = datasetFile
		self._data = Dataset.from_file(self._datasetFilename)
		self._imgWidth, self._imgHeight = self._data.imgSize

		# Setup transformation objects
		self._undistortTransform = UndistortionTransform(self._data.camera)
		self._pxTransform = PixelCoordinateTransform(self._data.imgSize)


	def initialize(self):
		""" Initialize new output folder and prepare pipeline for execution

		"""

		# Initialize output folders
		self._datasetOutputDir = self._outputDir + self._data.name
		if not os.path.exists(self._datasetOutputDir):
			os.makedirs(self._datasetOutputDir)

		self._runDir =  self._datasetOutputDir + '/' + time.strftime('%d_%m_%Y_%H_%M_%S/')
		if not os.path.exists(self._runDir):
			os.makedirs(self._runDir)

		self._trackDir =  f"{self._runDir}tracks/"
		if not os.path.exists(self._trackDir):
			os.makedirs(self._trackDir)

		# Initialize grid object for measurement filtering
		self._measurementGrid = Grid(self._imgWidth, self._imgHeight, 200, 100)
		self._detectionGrid = Grid(self._imgWidth, self._imgHeight, 200, 30)

		# Setup Track and Measurement databases
		self._tDB = TrackDB()
		self._mDB = MeasurementDB(self._measurementGrid, 5)

		# Setup Grid Detector
		self._gd = GridDetector.from_grid(self._detector, self._detectionGrid, self._detectionLimit, 50)

		# Detect Initial Features
		img, timestamp = self._data.read()
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		points = self._gd.detect(grayImg, self._data.mask)

		# Instantiate tracks 
		self._tDB.addNewTracks([Track.from_point(p, timestamp) for p in points])

		# Setup LKTracker with default params
		self._lk = LKOpticalFlowTracker(grayImg)

		# Setup GPR field approximator with default params
		self._gp = field_approx.gp.GPApproximator()

		# Initialize timestamps for pipeline control
		self._lastDetectionTime = timestamp
		self._lastTimestamp = timestamp

	def run(self):
		while(self._data.more()):
			# Load next image
			img, timestamp = self._data.read()

			print(f"Dataset {self._data.progress:.2f}% Processed, Current Timestamp: {timestamp:.3f}")

			# Convert image to grayscale for detection and tracking
			grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Get current track end points
			endPoints = np.asarray(self._tDB.getActiveEndpoints())
			# Attempt to track end points using LK optical flow
			newPoints = self._lk.trackPoints(endPoints, grayImg)
			# Update track end points with results from LK tracker
			self._tDB.updateActiveTracks(newPoints, timestamp)

			# If active tracks are low or it is time to run a detection, do so
			if (self._tDB.getNumActiveTracks() < self._desiredActiveTracks or timestamp - self._lastDetectionTime > self._detectionInterval):
				# Mask active track end points
				searchMask = np.copy(self._data.mask)
				for point in self._tDB.getActiveEndpoints():
					cv2.circle(searchMask, tuple(point), 5, 0, -1)

				detections = self._gd.detect(grayImg, searchMask)
				self._tDB.addNewTracks([Track.from_point(p, timestamp) for p in detections])
				self._lastDetectionTime = timestamp
				del searchMask

			self._lastTimestamp = timestamp

			del grayImg, img

	def saveApproximation(self, timestamp=None):
		if timestamp is None:
			timestamp = self._lastTimestamp

		self._mDB.clearMeasurements()
		tracks = self._tDB.getHistoricalTracks()

		print("Extracting and Filtering Measurements")
		unwarpedTracks = [self._undistortTransform.transformTrack(t) for t in tracks]
		tracksForMeasurement = [self._pxTransform.transformTrack(t) for t in unwarpedTracks]
		for t in tracksForMeasurement:
			self._mDB.addMeasurements(t.measureVelocity(minDist=0.0, scoring='composite'))

		print("Run Gaussian Process Regression")
		measurements = self._mDB.getMeasurements(measurementsPerCell=1)
		if len(measurements) > 0:
			self._gp.clearMeasurements()
			self._gp.addMeasurements(measurements)
			approxField = self._gp.approximate()
		else:
			print("Error: No Measurements Available")
			return

		print("Saving Approximation to File")

		approxFieldFile = f"{self._runDir}approx_{timestamp:.2f}.field"
		with open(approxFieldFile, mode='wb') as f:
			dill.dump(approxField, f)


	def saveTracks(self, timestamp=None):
		if timestamp is None:
			timestamp = self._lastTimestamp

		tracks = self._tDB.getAllTracks()

		for i, t in enumerate(tracks):
			t.save(self._trackDir + f"track_{i}.yaml")