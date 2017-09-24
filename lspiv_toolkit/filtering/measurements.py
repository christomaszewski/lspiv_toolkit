import numpy as np

import matplotlib.pyplot as plt

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

	def getMeasurements(self, measurementsPerCell=None):
		measurements = []


		if measurementsPerCell is None:
			measurementsPerCell = self._cellMeasurementLimit

		for mBin in self._measurementBins.values():

			# Only take as many measurements as are available
			bound = min(measurementsPerCell, len(mBin))
			numTaken = 0
			indexLastTaken = 0

			mID = set()

			while bound > numTaken:
				# Always take highest score measurement
				if (numTaken == 0):
					measurements.append(mBin[0])
					numTaken += 1
					indexLastTaken = 0
					mID.add(mBin[0].id)
				elif indexLastTaken < len(mBin)-1:
					# If we haven't sampled all unique tracks in the bucket yet
					startPoint = indexLastTaken + 1
					for i, m in enumerate(mBin[startPoint:]):
						if m.id not in mID:
							measurements.append(m)
							numTaken += 1
							indexLastTaken = startPoint + i
							mID.add(m.id)
							break
					# No more measurements from unique tracks in bin
					indexLastTaken = len(mBin)-1
				else:
					#just grab as many top scored measurements as we need
					# this may choose the same measurements more than once
					measurements.extend(mBin[1:(bound - numTaken + 1)])
					indexLastTaken = bound - numTaken
					numTaken = bound

		return measurements

	def getUniqueCoverage(self, measurementsPerCell=None):
		"""
			Iterate through all measurement bins and try to collect top
			x unique measurement ids per bin where x is measurementsPerCell
		"""
		coverage = set()

		if measurementsPerCell is None:
			measurementsPerCell = self._cellMeasurementLimit

		for mBin in self._measurementBins.values():

			# Only take as many measurements as are available
			bound = min(measurementsPerCell, len(mBin))
			numTaken = 0
			indexLastTaken = -1
			
			while (indexLastTaken < len(mBin)-1) and bound > numTaken:
				indexLastTaken += 1
				mID = mBin[indexLastTaken].id
				if mID not in coverage:
					coverage.add(mID)
					numTaken += 1

		return coverage

	def getCoverage(self, measurementsPerCell=None):
		"""
			Iterate through all measurement bins and try to collect 
			top x measurement ids per bin where x is measurementsPerCell.
			Does not care if id was already collected in a previous bin 
			or if multiple ids chosen in each bin are the same
		"""
		coverage = set()

		if measurementsPerCell is None:
			measurementsPerCell = self._cellMeasurementLimit

		for mBin in self._measurementBins.values():

			# Only take as many measurements as are available
			bound = min(measurementsPerCell, len(mBin))
			for m in mBin[:bound]:
				coverage.add(m.id)

		return coverage

	def getBinnedScores(self):
		binScore = defaultdict(list)

		for key in self._measurementBins:
			binScore[key] = [-m.score for m in self._measurementBins[key]]

		return binScore