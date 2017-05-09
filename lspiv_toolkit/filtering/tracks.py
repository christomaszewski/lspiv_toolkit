import heapq

from primitives.track import Track, TrackState

class TrackDB(object):

	def __init__(self):
		# Threshold for moving tracks to historical db
		self._historicalThreshold = 0.004 # Allows 1 frame drop with 30 fps video

		# Minimum age of historical tracks
		self._minAge = 3 #seconds

		# Minimum total displacement of historical tracks
		self._minDisplacement = 10 #pixels

		# Minimum avg speed of historical tracks
		self._minSpeed = 1 #px/s

		# Min Meandering ratio: displacement/(avgSpeed * age)
		self._minMeadering = 0.9

		# List of active tracks
		self._activeList = []

		# Priority queue of historical tracks
		self._historicalPQ = []

	def getActiveTracks(self):
		return self._activeList

	def getNumActiveTracks(self):
		return len(self._activeList)

	def getHistoricalTracks(self):
		return self._historicalPQ

	def getAllTracks(self):
		tracks = list(self._activeList)
		tracks.extend(self._historicalPQ)
		return tracks

	def getActiveEndpoints(self):
		endpoints = [t.endPoint for t in self._activeList]
		return endpoints

	def addNewTrack(self, track):
		#heapq.heappush(self._activePQ, track)
		self._activeList.append(track)

	def addNewTracks(self, tracks):
		self._activeList.extend(tracks)


	def updateActiveTracks(self, points, timestamp):
		#todo: check that length of points is same as number of active tracks

		updatedTracks = []

		for i in range(0, len(self._activeList)-1):
			t = self._activeList[i]

			if (points[i] is not None):
				t.addObservation(tuple(points[i]), timestamp)
				t.state = TrackState.ACTIVE
				updatedTracks.append(t)
				#heapq.heappush(updatedTracks, t)
			else:
				if (timestamp - t.lastSeen > self._historicalThreshold):
					# Candidate for storage
					age = t.age()
					displacement = t.length()
					if (age < self._minAge):
						continue
					elif (displacement < self._minDisplacement):
						print('track too short')
					else:
						print('moving track to historical db', age, displacement)
						t.state = TrackState.HISTORICAL
						heapq.heappush(self._historicalPQ, t)
				else:
					t.state = TrackState.LOST
					#heapq.heappush(updatedTracks, t)
					updatedTracks.append(t)


		self._activeList = updatedTracks

		print("active tracks:", len(self._activeList))
		print("historical tracks:", len(self._historicalPQ))

		if (len(self._historicalPQ) > 100):
			self.pruneTracks()


	def pruneTracks(self, numTracks=50):
		prunedTracks = list(self._historicalPQ[:numTracks])

		heapq.heapify(prunedTracks)

		self._historicalPQ = prunedTracks