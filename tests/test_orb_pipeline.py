import cv2
import numpy as np

from primitives.grid import Grid

from context import lspiv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ORBDetector
from cv_toolkit.detect.adapters import KeyPointGridDetector

from cv_toolkit.track.features import SparseFeatureTracker

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
	datasetFilename = "../../../datasets/llobregat_short.yaml"
	data = Dataset.from_file(datasetFilename)

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

	orbParams = dict(nfeatures=100, nlevels=1)

	detector = ORBDetector(orbParams)

	gd = KeyPointGridDetector(detector, (20, 30), 100000, 0)
	transformation = UndistortionTransform(data.camera)

	sfTracker = SparseFeatureTracker(gd)

	timestamp = data.currentTime()
	img = data.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	points = sfTracker.initialize(gray, data.mask)
	lastDetectionTime = timestamp

	# Instantiate tracks 
	for p in points:
		t = Track.from_key_point(p, timestamp)
		tDB.addNewTrack(t)

	cv2.namedWindow('img', cv2.WINDOW_NORMAL)

	loopcount = 0


	#transformation = IdentityTransform()
	transformation = UndistortionTransform(data.camera)

	while(data.more()):
		timestamp = data.currentTime()
		img = data.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		undistortedImg = transformation.transformImage(img)

		tView.imshow(undistortedImg)

		points = np.asarray(tDB.getActiveKeyPoints())

		newPoints, indices = sfTracker.trackPoints(points, gray, data.mask)

		tDB.updateActiveTracksKeyPoints(newPoints, timestamp, indices)

		points = np.asarray([np.squeeze(p.point) for p in newPoints if p is not None])

		if (tDB.getNumActiveTracks() < desiredActiveTracks or timestamp - lastDetectionTime > detectionInterval):
			# Mask active track end points
			searchMask = np.copy(data.mask)

			print("detecting...", timestamp)

			for point in tDB.getActiveEndpoints():
				cv2.circle(searchMask, tuple(point.astype('int32')), 5, 0, -1)

			detections = []# gd.detect(gray, searchMask)
			print("detected new features")

			for p in detections:
				t = Track.from_point(p, timestamp)
				tDB.addNewTrack(t)

			lastDetectionTime = timestamp
			print("added new tracks")


		loopcount += 1
		print(loopcount)

		warpedActiveTracks = [transformation.transformTrack(t) for t in tDB.getActiveTracks()]
		warpedHistoricalTracks = [transformation.transformTrack(t) for t in tDB.getHistoricalTracks()]
		tView.drawEndPoints(warpedActiveTracks)
		tView.drawTracks(warpedHistoricalTracks)

		if (cv2.waitKey(1) & 0xFF) == 27:
			break

	cv2.destroyAllWindows()

if __name__ == '__main__':
	mainFunc()