from sortedcontainers import SortedList

from primitives.track import Track, TrackState

class TrackDB(object):

	def __init__(self):
		# Threshold for moving tracks to historical db
		self._historicalThreshold = 0.004 # Allows 1 frame drop with 30 fps video

		# Minimum age of historical tracks
		self._minAge = 1.5 #seconds

		# Minimum total displacement of historical tracks
		self._minDisplacement =  60 #pixels

		# Minimum avg speed of historical tracks
		self._minSpeed = 1 #px/s

		# Min Meandering ratio: displacement/(avgSpeed * age)
		self._minMeadering = 0.9

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
				if (timestamp - t.lastSeen > self._historicalThreshold):
					# Candidate for storage
					age = t.age()
					displacement = t.length()
					if (age < self._minAge):
						del t
						continue
					elif (displacement < self._minDisplacement):
						print('track too short')
						del t
					elif (t.avgSpeedFast < self._minSpeed):
						print('track too slow')
						del t
					else:
						print('moving track to historical db', age, displacement)
						t.state = TrackState.HISTORICAL
						self._historicalList.add(t)
				else:
					t.state = TrackState.LOST
					updatedTracks.append(t)


		self._activeList = updatedTracks

		print("active tracks:", len(self._activeList))
		print("historical tracks:", len(self._historicalList))

	def updateActiveTracks(self, points, timestamp):
		#todo: check that length of points is same as number of active tracks

		updatedTracks = []
		activeList = self._activeList

		for i, t in enumerate(activeList):

			if (points[i] is not None):
				if (points[i].ndim < 1):
					print("error in updates:", points[i])
				t.addObservation(points[i], timestamp)
				t.state = TrackState.ACTIVE
				updatedTracks.append(t)
				#heapq.heappush(updatedTracks, t)
			else:
				if (timestamp - t.lastSeen > self._historicalThreshold):
					# Candidate for storage
					age = t.age()
					displacement = t.length()
					if (age < self._minAge):
						del t
						continue
					elif (displacement < self._minDisplacement):
						#print('track too short')
						del t
					elif (t.avgSpeedFast < self._minSpeed):
						#print('track too slow')
						del t
					else:
						#print('moving track to historical db', age, displacement)
						t.state = TrackState.HISTORICAL
						self._historicalList.add(t)
				else:
					t.state = TrackState.LOST
					updatedTracks.append(t)


		del self._activeList
		self._activeList = updatedTracks

		print("active tracks:", len(self._activeList))
		print("historical tracks:", len(self._historicalList))

		if (len(self._historicalList) > 2000):
			self.pruneTracks()


	def pruneTracks(self, numTracks=1000):
		self._historicalList = SortedList(self._historicalList[:numTracks])

		