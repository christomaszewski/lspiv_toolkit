import cv2
import numpy as np

from context import lspiv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ShiTomasiDetector
from cv_toolkit.detect.adapters import GridDetector

from cv_toolkit.track.flow import LKOpticalFlowTracker

from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import IdentityTransform

import field_toolkit.approx as field_approx
import field_toolkit.viz as field_viz

from primitives.track import Track
from primitives.grid import Grid

from lspiv_toolkit.filtering.tracks import TrackDB
from lspiv_toolkit.filtering.measurements import MeasurementDB
from lspiv_toolkit.viz.plotting import TrackView

from memory_profiler import profile

@profile
def mainFunc():

	# Load dataset from file
	datasetFilename = "../../../datasets/llobregat_short.yaml"
	d = Dataset.from_file(datasetFilename)

	# Setup Track Database
	tDB = TrackDB()

	# Setup image display
	tView = TrackView('img')

	# Pipeline parameters
	detectionInterval = 1 #seconds
	approximationInterval = 5 #seconds
	desiredActiveTracks = 300
	initialDetection = 30001

	# Initialize grid object for measurement filtering
	g = Grid(3840, 2160, 100, 50)

	# Setup measurement database
	mDB = MeasurementDB(g, 5)

	# Instantiate detector
	detector = ShiTomasiDetector()

	# Setup Grid Detector
	gd = GridDetector(detector, (20, 300), initialDetection, 0)

	# Detect Initial Features
	timestamp = d.currentTime()
	lastApproximated = timestamp
	img = d.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	points = gd.detect(gray, d.mask)
	lastDetectionTime = timestamp

	# Instantiate tracks 
	for p in points:
		t = Track.from_point(p, timestamp)
		tDB.addNewTrack(t)

	gd.setGrid((6,100))
	gd.featureLimit = 2*desiredActiveTracks + 1

	# Setup LKTracker
	winSize = (15,15)
	maxLevel = 0 # pyramids currently leak memory
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

	lk = LKOpticalFlowTracker(winSize, maxLevel, criteria, gray)

	gp = field_approx.gp.GPApproximator()

	#transformation = IdentityTransform()
	transformation = UndistortionTransform(d.camera)

	while(d.more()):
		timestamp = d.currentTime()
		img = d.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		undistImg = transformation.transformImage(img)
		#points = np.float32(points).reshape(-1, 2)

		tView.imshow(undistImg)

		points = np.asarray(tDB.getActiveEndpoints())

		points = lk.trackPoints(points, gray)

		tDB.updateActiveTracks(points, timestamp)

		if (tDB.getNumActiveTracks() < desiredActiveTracks or timestamp - lastDetectionTime > detectionInterval):
			# Mask active track end points
			searchMask = np.copy(d.mask)

			print("detecting...", timestamp)

			for point in tDB.getActiveEndpoints():
				cv2.circle(searchMask, tuple(point), 5, 0, -1)

			detections = gd.detect(gray, searchMask)
			print("detected new features")

			for p in detections:
				t = Track.from_point(p, timestamp)
				tDB.addNewTrack(t)

			lastDetectionTime = timestamp
			print("added new tracks")


		if (timestamp - lastApproximated > approximationInterval):
			print("approximating")

			tracks = sorted(tDB.getAllTracks())
			print(tracks[0].score)
			print(tracks[-1].score)

			# Take top 100 Tracks
			if (len(tracks) > 100):
				tracks = tracks[:100]

			print("filtering measurements")
			for t in tracks:
				mDB.addMeasurements(t.measureVelocity(scoring='composite'))

			print("running gpr")
			gp.clearMeasurements()
			gp.addMeasurements(mDB.getMeasurements())
			#gp.approximate()

			lastApproximated = timestamp
		"""
		hist = list(tDB.getHistoricalTracks())
		act = list(tDB.getActiveTracks())

		bestTracks = []

		for i in range(0,4):
			if (len(hist) > 0):
				bestTracks.append(heapq.heappop(hist))
			if (len(act) > 0):
				bestTracks.append(heapq.heappop(act))

		tView.drawTracks(bestTracks)
		"""


		warpedActiveTracks = [transformation.transformTrack(t) for t in tDB.getActiveTracks()]
		warpedHistoricalTracks = [transformation.transformTrack(t) for t in tDB.getHistoricalTracks()]
		tView.drawEndPoints(warpedActiveTracks)
		tView.drawTracks(warpedHistoricalTracks)

		if (cv2.waitKey(1) & 0xFF) == 27:
			break

	cv2.destroyAllWindows()

if __name__ == '__main__':
	mainFunc()