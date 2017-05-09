import cv2
import numpy as np
import heapq

from cv_toolkit.data import Dataset
from cv_toolkit.track import LKOpticalFlowTracker
from cv_toolkit.detect import GridDetector, ShiTomasiDetector

import field_toolkit.approx as field_approx
import field_toolkit.viz as field_viz

from primitives.track import Track
from primitives.grid import Grid

from filtering.tracks import TrackDB
from filtering.measurements import MeasurementDB
from viz.plotting import TrackView

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

# Setup Grid Detector
gd = GridDetector.using_shi_tomasi((20, 300), initialDetection, 0)

# Detect Initial Features
timestamp = d.currentTime()
lastApproximated = timestamp
img = d.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
points = gd.detect(gray, d.mask)
lastDetectionTime = timestamp

# Instantiate tracks 
for p in points:
	position = tuple(p)
	t = Track(position, timestamp)
	tDB.addNewTrack(t)

gd.setGrid((6,100))
gd.featureLimit = 2*desiredActiveTracks + 1

# Setup LKTracker
winSize = (15,15)
maxLevel = 5
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

lk = LKOpticalFlowTracker(winSize, maxLevel, criteria, gray)

gp = field_approx.gp.GPApproximator()


while(d.more()):
	timestamp = d.currentTime()
	img = d.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#points = np.float32(points).reshape(-1, 2)

	tView.imshow(img)

	points = np.float32(tDB.getActiveEndpoints()).reshape(-1,2)

	points = lk.trackPoints(points, gray)

	tDB.updateActiveTracks(points, timestamp)

	if (tDB.getNumActiveTracks() < desiredActiveTracks or timestamp - lastDetectionTime > detectionInterval):
		# Mask active track end points
		searchMask = np.copy(d.mask)

		for point in tDB.getActiveEndpoints():
				cv2.circle(searchMask, point, 5, 0, -1)

		detections = gd.detect(gray, searchMask)

		for p in detections:
			position = tuple(p)
			t = Track(position, timestamp)
			tDB.addNewTrack(t)

		lastDetectionTime = timestamp
		print("detected new features")


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
		gp.approximate()

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
	tView.drawEndPoints(tDB.getActiveTracks())
	tView.drawTracks(tDB.getHistoricalTracks())

	cv2.waitKey(1)