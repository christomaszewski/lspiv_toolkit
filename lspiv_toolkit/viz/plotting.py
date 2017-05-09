import matplotlib.pyplot as plt
import cv2
import numpy as np

from primitives.track import TrackState

plt.ion()

class TrackView(object):

	def __init__(self, windowName, image=None):
		self._name = windowName
		self._img = image

		cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

		self.refresh()

	def refresh(self):
		if (self._img is not None):
			cv2.imshow(self._name, self._img)


	def imshow(self, image):
		self._img = image

		self.refresh()

	def drawEndPoint(self, track, color=(255,0,0)):
		if (track.state is TrackState.LOST):
			color = (0, 0, 255)

		cv2.circle(self._img, track.endPoint, 3, color, -1)

	def drawTrack(self, track, color=(255,0,0)):
		points = np.int32(np.asarray(track.positions).reshape(-1,2))

		cv2.polylines(self._img, [points], True, color)

	def drawEndPoints(self, tracks):
		if len(tracks) == 0:
			return

		scores = [-t.score for t in tracks]
		maxScore = max(scores)
		for t in tracks:
			score = int((-t.score/maxScore)*255)
			color = (255-score, score, 0)
			if (t.state is TrackState.LOST):
				color = (0, 0, 255)
			self.drawEndPoint(t, color)

		self.refresh()

	def drawTracks(self, tracks):
		if len(tracks) == 0:
			return

		scores = [-t.score for t in tracks]
		maxScore = max(scores)
		for t in tracks:
			score = int((-t.score/maxScore)*255)
			color = (255-score, score, 0)
			if (t.state is TrackState.LOST):
				color = (0, 0, 255)
			self.drawTrack(t, color)

		self.refresh()