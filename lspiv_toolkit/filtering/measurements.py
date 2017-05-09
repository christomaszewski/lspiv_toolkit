import numpy as np
import heapq

from collections import defaultdict
from sortedcontainers import SortedList

from primitives.measurement import Measurement
from primitives.grid import Grid

class MeasurementDB(object):

	def __init__(self, grid, cellMeasurementLimit):
		self._grid = grid
		self._cellMeasurementLimit = cellMeasurementLimit

		self._measurementBins = defaultdict(SortedList)

	def addMeasurements(self, measurements):

		for m in measurements:
			self.addMeasurement(m)

	def addMeasurement(self, measurement):
		gridCoord = self._grid.bin(measurement.point)

		bucket = self._measurementBins[gridCoord]

		if (len(bucket) >= self._cellMeasurementLimit):
			bucket.add(measurement)
			bucket.pop()
			#heapq.heappushpop(bucket, measurement)
		else:
			bucket.add(measurement)
			#heapq.heappush(bucket, measurement)

	def clearMeasurements(self):
		self._measurementBins.clear()

	def getMeasurements(self):
		measurements = []

		for key in self._measurementBins:
			measurements.extend(self._measurementBins[key])

		return measurements
