from sortedcontainers import SortedList

from primitives.track import Track, TrackState

class TrackDB(object):

	def __init__(self, threshold=0.004, minAge=1.55, minDisplacement=100, minSpeed=1, meanderingRatio=0.9):
		# Threshold for moving tracks to historical db
		self._historicalThreshold = threshold # 0.004 Allows 1 frame drop with 30 fps video

		# Minimum age of historical tracks
		self._minAge = minAge #seconds

		# Minimum total displacement of historical tracks
		self._minDisplacement = minDisplacement #pixels

		# Minimum avg speed of historical tracks
		self._minSpeed = minSpeed #px/s

		# Min Meandering ratio: displacement/(avgSpeed * age)
		self._meanderingRatio = meanderingRatio

		print(threshold, minAge, minDisplacement, minSpeed, meanderingRatio)

		# List of active tracks
		self._activeList = []

		# SortedList of historical tracks
		self._historicalList = SortedList()

	def getActiveTracks(self):
		return self._activeList

	def getNumActiveTracks(self):
		return len(self._activeList)

	def getHistoricalTracks(self):
		return self._historicalList

	def getAllTracks(self):
		tracks = list(self._activeList)
		tracks.extend(self._historicalList)
		return tracks

	def getActiveEndpoints(self):
		endpoints = [t.endPoint for t in self._activeList]
		return endpoints

	def getActiveKeyPoints(self):
		keyPoints = [t.lastKeyPoint for t in self._activeList]
		return keyPoints

	def addNewTrack(self, track):
		self._activeList.append(track)

	def addNewTracks(self, tracks):
		self._activeList.extend(tracks)

	def updateActiveTracksKeyPoints(self, keyPoints, timestamp, indices):
		updatedTracks = []
		activeList = self._activeList

		for i, index in enumerate(indices):
			t = activeList[index]
			if (keyPoints[i] is not None):
				t.addKeyPointObservation(keyPoints[i], timestamp)
				t.state = TrackState.ACTIVE
				updatedTracks.append(t)
			else:
				if (timestamp - t.lastSeen > self._historicalThreshold and t.size() > 1):
					# Candidate for storage
					age = t.age()
					displacement = t.displacement()
					distance = t.distance()
					meanderingRatio = displacement / distance
					if (age < self._minAge):
						del t
					elif (displacement < self._minDisplacement):
						print('track too short')
						del t
					elif (t.avgSpeedFast < self._minSpeed):
						print('track too slow')
						del t
					elif (meanderingRatio < self._meanderingRatio):
						print('track meanders')
						del t
					else:
						print('moving track to historical db', age, displacement)
						t.state = TrackState.HISTORICAL
						self._historicalList.add(t)
				else:
					t.state = TrackState.LOST
					updatedTracks.append(t)


		self._activeList = updatedTracks

		print(f"Updating tracks. Active: {len(self._activeList)}, Historical: {len(self._historicalList)}")
		#print("active tracks:", len(self._activeList))
		#print("historical tracks:", len(self._historicalList))

	def updateActiveTracks(self, points, timestamp):
		#todo: check that length of points is same as number of active tracks
		activeList = self._activeList
		del self._activeList
		self._activeList = []

		for i, t in enumerate(activeList):

			if (points[i] is not None):
				if (points[i].ndim < 1):
					print("error in updates:", points[i])
				t.addObservation(points[i], timestamp)
				#t.state = TrackState.ACTIVE
				self._activeList.append(t)
				#heapq.heappush(self._activeList, t)
			else:
				if (timestamp - t.lastSeen > self._historicalThreshold and t.size() > 1):
					# Candidate for storage
					age = t.age()
					displacement = t.displacement()
					distance = t.distance()
					meanderingRatio = displacement / distance
					if (age < self._minAge):
						#print('track age too short')
						del t
					elif (displacement < self._minDisplacement):
						#print('track displacement too short')
						del t
					elif (t.avgSpeedFast < self._minSpeed):
						#print('track too slow')
						del t
					elif (meanderingRatio < self._meanderingRatio):
						#print('track meanders')
						del t
					else:
						#print('moving track to historical db', age, displacement)
						t.state = TrackState.HISTORICAL
						self._historicalList.add(t)
				else:
					t.state = TrackState.LOST
					self._activeList.append(t)

		del activeList

		print(f"Updating tracks. Active: {len(self._activeList)}, Historical: {len(self._historicalList)}")

		if (len(self._historicalList) > 20000):
			self.pruneTracks()


	def pruneTracks(self, numTracks=15000):
		del self._historicalList[numTracks:]