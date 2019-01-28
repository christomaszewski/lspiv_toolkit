import yaml
import os

# High Priority!
# Configure pipelines, config files for output
# files for how output was generated maybe pipeline config is enough here

# Maybe there should be a separate config for
# detection, measurement filtering, and track filtering

class ApproximationConfig(yaml.YAMLObject):
	yaml_tag = '!Approximation_Config'

	def __init__(self, inputDir, cameraFile=None):
		self._inputDir = os.path.abspath(inputDir)
		self._trainingSets = ['raw']

		# Camera file for unwarping
		if cameraFile is None:
			self._camFile = f"{inputDir}/camera.yaml"
		else:
			self._camFile = cameraFile

		# Relevant to measurement extraction from tracks
		self._measurementMethod = 'savitzkyGolayPreFilter'
		self._measurementMethodParams = {'windowSize':45, 'order':3}
		self._scoringMethod = 'time'

		# Relevant to measurement filtering
		self._measurementGridDim = (240, 108)
		self._measurementBinCapacity = 100
		self._measurementsPerCell = 1
		self._filteringMethod = 'max'

		# Relevant to gp reconstruction
		self._approximationMethod = 'simple'
		# Eventually would like to add some kernel stuff here

	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			return yaml.load(f)

	def save(self, filename=None):
		if (filename is None):
			filename = f"{self._inputDir}/approx_config.yaml"

		with open(filename, mode='w') as f:
			yaml.dump(self, f)

	def getFilteringParams(self):
		params = {'measurementBinCapacity': self._measurementBinCapacity,
					'filteringMethod':self._filteringMethod}
		return params

	def getMeasurementParams(self):
		params = {'method':self._measurementMethod, 'scoring':self._scoringMethod}
		return params

	@property
	def inputDir(self):
		return self._inputDir

	@inputDir.setter
	def inputDir(self, directory):
		self._inputDir = directory

	@property
	def trainingSets(self):
		return self._trainingSets

	@trainingSets.setter
	def trainingSets(self, directoryList):
		self._trainingSets = directoryList
	
	@property
	def camFile(self):
		return self._camFile

	@camFile.setter
	def camFile(self, file):
		self._camFile = file

	@property
	def measurementMethod(self):
		return self._measurementMethod

	@measurementMethod.setter
	def measurementMethod(self, method):
		self._measurementMethod = method

	@property
	def measurementMethodParams(self):
		return self._measurementMethodParams

	@measurementMethodParams.setter
	def measurementMethodParams(self, params):
		self._measurementMethodParams = params

	@property
	def measurementGridDim(self):
		return self._measurementGridDim

	@measurementGridDim.setter
	def measurementGridDim(self, dimensions):
		self._measurementGridDim = dimensions

	@property
	def measurementBinCapacity(self):
		return self._measurementBinCapacity

	@measurementBinCapacity.setter
	def measurementBinCapacity(self, capacity):
		self._measurementBinCapacity = capacity

	@property
	def measurementsPerCell(self):
		return self._measurementsPerCell

	@measurementsPerCell.setter
	def measurementsPerCell(self, measurementLimit):
		self._measurementsPerCell = measurementLimit

	@property
	def filteringMethod(self):
		return self._filteringMethod

	@filteringMethod.setter
	def filteringMethod(self, method):
		self._filteringMethod = method

	@property
	def scoringMethod(self):
		return self._scoringMethod

	@scoringMethod.setter
	def scoringMethod(self, method):
		self._scoringMethod = method

	@property
	def approximationMethod(self):
		return self._approximationMethod

	@approximationMethod.setter
	def approximationMethod(self, method):
		self._approximationMethod = method
