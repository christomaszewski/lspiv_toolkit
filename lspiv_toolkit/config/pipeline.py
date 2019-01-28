import yaml
import os

class PipelineConfig(yaml.YAMLObject):
	yaml_tag = '!Pipeline_Config'

	def __init__(self, data, output):
		self._datasetFile = os.path.abspath(data)
		self._outputDir = os.path.abspath(output)

		# Feature Detection Settings
		self._detectionGridDim = (400, 30)
		self._numDesiredTracks = 1000
		self._detectionInterval = 1
		self._maxFeatures = 60001
		self._borderBuffer = 50
		# Need to add detector type 
		self._maxCellFeatures = int(self._maxFeatures/(self._detectionGridDim[0] * self._detectionGridDim[1]))
		self._qualityLevel = 0.3
		self._minFeatureDistance = 10.
		self._blockSize = 10

		# Track Filtering settings
		self._historicalThreshold = 0.004
		self._minAge = 1.55
		self._minDisplacement = 100.
		self._minSpeed = 1.0
		self._meanderingRatio = 0.9
		self._maxTracks = 20000

		# LK Settings
		self._windowSize = (21,21)
		self._maxLevel = 5
		self._maxIter = 30
		self._epsilon = 0.01

	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			return yaml.load(f)

	def save(self, filename=None):
		if (filename is None):
			filename = f"{self._outputDir}/config.yaml"

		with open(filename, mode='w') as f:
			yaml.dump(self, f)

	def getTrackFilteringParams(self):
		params = {'threshold':self._historicalThreshold, 'minAge':self._minAge,
				'minDisplacement':self._minDisplacement, 'minSpeed':self._minSpeed,
				'meanderingRatio':self._meanderingRatio, 'maxTracks':self._maxTracks}
		return params

	def getFeatureDetectionParams(self):
		params = {'maxCorners':self._maxCellFeatures, 'qualityLevel':self._qualityLevel,
				'minDistance':self._minFeatureDistance, 'blockSize':self._blockSize}
		return params

	def getLKFlowParams(self):
		params = {'winSize':self._windowSize, 'maxLevel':self._maxLevel,
				'maxIter':self._maxIter, 'epsilon':self._epsilon}
		return params

	@property
	def datasetFile(self):
		return self._datasetFile

	@datasetFile.setter
	def datasetFile(self, filename):
		self._datasetFile = os.path.abspath(filename)

	@property
	def outputDir(self):
		return self._outputDir

	@outputDir.setter
	def outputDir(self, outputDir):
		self._outputDir = os.path.abspath(outputDir)

	@property
	def detectionGridDim(self):
		return self._detectionGridDim

	@detectionGridDim.setter
	def detectionGridDim(self, dimensions):
		self._detectionGridDim = dimensions

	@property
	def numDesiredTracks(self):
		return self._numDesiredTracks

	@numDesiredTracks.setter
	def numDesiredTracks(self, numTracks):
		self._numDesiredTracks = numTracks

	@property
	def detectionInterval(self):
		return self._detectionInterval

	@detectionInterval.setter
	def detectionInterval(self, interval):
		self._detectionInterval = interval

	@property
	def maxFeatures(self):
		return self._maxFeatures

	@maxFeatures.setter
	def maxFeatures(self, numFeatures):
		self._maxFeatures = numFeatures

	@property
	def borderBuffer(self):
		return self._borderBuffer

	@borderBuffer.setter
	def borderBuffer(self, borderBuffer):
		self._borderBuffer = borderBuffer

	@property
	def maxCellFeatures(self):
		return self._maxCellFeatures

	@maxCellFeatures.setter
	def maxCellFeatures(self, numFeatures):
		self._maxCellFeatures = numFeatures

	@property
	def qualityLevel(self):
		return self._qualityLevel

	@qualityLevel.setter
	def qualityLevel(self, quality):
		self._qualityLevel = quality

	@property
	def minFeatureDistance(self):
		return self._minFeatureDistance

	@minFeatureDistance.setter
	def minFeatureDistance(self, distance):
		self._minFeatureDistance = distance

	@property
	def blockSize(self):
		return self._blockSize

	@blockSize.setter
	def blockSize(self, size):
		self._blockSize = size

	@property
	def historicalThreshold(self):
		return self._historicalThreshold

	@historicalThreshold.setter
	def historicalThreshold(self, threshold):
		self._historicalThreshold = threshold

	@property
	def minAge(self):
		return self._minAge

	@minAge.setter
	def minAge(self, age):
		self._minAge = age

	@property
	def minDisplacement(self):
		return self._minDisplacement

	@minDisplacement.setter
	def minDisplacement(self, displacement):
		self._minDisplacement = displacement

	@property
	def meanderingRatio(self):
		return self._meanderingRatio

	@meanderingRatio.setter
	def meanderingRatio(self, ratio):
		self._meanderingRatio = ratio

	@property
	def maxTracks(self):
		return self._maxTracks

	@maxTracks.setter
	def maxTracks(self, maxTracks):
		self._maxTracks = maxTracks


	@property
	def windowSize(self):
		return self._windowSize

	@windowSize.setter
	def windowSize(self, size):
		self._windowSize = size

	@property
	def maxLevel(self):
		return self._maxLevel

	@maxLevel.setter
	def maxLevel(self, level):
		self._maxLevel = level

	@property
	def maxIter(self):
		return self._maxIter

	@maxIter.setter
	def maxIter(self, iterations):
		self._maxIter = iterations

	@property
	def epsilon(self):
		return self._epsilon

	@epsilon.setter
	def epsilon(self, epsilon):
		self._epsilon = epsilon