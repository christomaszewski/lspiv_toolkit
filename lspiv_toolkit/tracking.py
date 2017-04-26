import numpy as np

from field_toolkit.core.primitives import Measurement

class Track(object):
	""" Represents a single particle/point on an object tracked over some 
		period of time. Can be used to produce vector field measurements

		self._positions holds particle locations in 2-Space
		self._times holds the times the particle was seen at the location

	"""

	def __init__(self, position=None, time=None):
		self._positions = []
		self._times = []

		if (position is not None):
			self._positions.append(position)
			if (time is None):
				self._times.append(time.time())
			else:
				self._times.append(time)

	def __getitem__(self, index):
		""" Overrides [] operator to return the observation at index

			Returns observation:
			(time, position) where position is (x,y) tuple

		"""
		if (index >= len(self._positions)):
			# Index out of bounds
			return None

		return (self._times[index], self._positions[index])

	def addObservation(self, position, time=None):
		if (time is None):
			time = time.time()

		self._positions.append(position)
		self._times.append(time)

	def addObservations(self, position, time):
		self._positions.extend(position)
		self._times.extend(time)

	def getLastObservation(self):
		""" Currently just returns the position of the last observation

		"""
		if (len(self._positions) < 1):
			return (None, None)

		return (self._times[-1], self._positions[-1])

	def size(self):
		return len(self._positions)

	def age(self):
		return self._time()

	def getPointSequence(self):
		# Todo: change return format for easy plotting
		return self._positions

	def updatePointSequence(self, points):
		""" For used in applying bulk transforms to all points in track

		"""
		self._positions = points

	def smoothTrack(self, smoothingFactor=3):
		newPoints = []


	def getMeasurements(self, method='midpoint', scoring='time'):
		""" Returns list of measurements representing velocity of particle
			localizing the measurement using the method specified. Velocity
			is computed by comparing pairs on consecutive points.

			midpoint: localize the measurement on the midpoint of the segment 
			between two consecutive particle locations
			front: localize measurement on first point of consecutive point pairs
			end: localize measurement on second point of consecutive point pairs

			Should return empty list of measurements if 0 or 1 observations

			Todo: Don't recalculate measurements if track has not changed since last
			function call
		"""
		methodFuncName = "_" + method
		methodFunc = getattr(self, methodFuncName, lambda p1, p2: p1)
		scoringFuncName = "_" + scoring
		scoringFunc = getattr(self, scoringFuncName, lambda: 0.0)

		score = scoringFunc()

		measurements = []

		prevPoint = None
		prevTime = None

		for timestamp, point in zip(self._times, self._positions):
			if prevPoint is not None:
				deltaT = timestamp - prevTime
				#print(point, prevPoint)

				xVel = (point[0] - prevPoint[0]) / deltaT
				yVel = (point[1] - prevPoint[1]) / deltaT
				vel = (xVel, yVel)

				measurementPoint = methodFunc(prevPoint, point)

				m = Measurement(measurementPoint, vel, score)
				measurements.append(m)

			prevPoint = point
			prevTime = timestamp

		return measurements

	# Method Functions
	def _first(self, p1, p2):
		return p1

	def _last(self, p1, p2):
		return p2

	def _midpoint(self, p1, p2):
		x = (p1[0] + p2[0]) / 2
		y = (p1[1] + p2[1]) / 2
		return (x, y)
	
	# Scoring Functinns
	def _time(self):
		# Length of track in time
		return self._times[-1] - self._times[0]

	def _length(self):
		# Length of track in number of measurements
		return self.size()

	def _constant(self):
		return 999999

	@property
	def positions(self):
		return self._positions

	@property
	def times(self):
		return self._times

	def __sub__(self, other):
		# Only subtracts matching times
		differences = []
		for t1, pt1, t2, pt2 in zip(self._times, self._positions, other.times, other.positions):
			if (t1 == t2):
				differences.append((pt1[0]-pt2[0], pt1[1]-pt2[1]))

		return differences

	def __rsub__(self, other):
		differences = []
		for t1, pt1, t2, pt2 in zip(other.times, other.positions, self._times, self._positions):
			if (t1 == t2):
				differences.append((pt1[0]-pt2[0], pt1[1]-pt2[1]))

		return differences