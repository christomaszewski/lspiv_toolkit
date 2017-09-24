import sys
import glob
import os
from scipy.interpolate import interp1d
import numpy as np

from context import lspiv_toolkit

from primitives.track import Track
from primitives.grid import Grid

import field_toolkit.approx as field_approx
from field_toolkit.core.fields import VectorField
from field_toolkit.analysis.drift import DriftAnalysis

from viz_toolkit.view import OverlayView, FieldOverlayView

from cv_toolkit.data import Dataset
from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform

from lspiv_toolkit.config import PipelineConfig, ApproximationConfig
from lspiv_toolkit.filtering.measurements import MeasurementDB

# Parse output directory
outputDir = os.path.abspath(sys.argv[1])

# Load pipeline config
configFile = f"{outputDir}/config.yaml"
config = PipelineConfig.from_file(configFile)

# Load dataset as specified in config file
data = Dataset.from_file(config.datasetFile)

# Setup transformation objects
unTrans = UndistortionTransform(data.camera)
pxTrans = PixelCoordinateTransform(data.imgSize)

# Load tracks
trackDir = f"{outputDir}/tracks"
trackFiles = glob.glob(f"{trackDir}/simple_coverage/track_*.yaml")
tracks = Track.from_file_list(trackFiles)


# Setup partition boundaries along x axis
boundaries = [0, 500, 1000, 1675, 2150, 2800, 3300, 3840]
trackFolders = [f"{trackDir}/{boundaries[i]}-{boundaries[i+1]}" for i in range(len(boundaries)-1)]
trackPartitions = {folder: [] for folder in trackFolders}

for dirName in trackFolders:
	if not os.path.exists(dirName):
		os.makedirs(dirName)

for t in tracks:
	transTrack = pxTrans.transformTrack(unTrans.transformTrack(t))
	startTime, startPoint = transTrack.getFirstObservation()
	endTime, endPoint = transTrack.getLastObservation()

	startBin = np.searchsorted(boundaries, startPoint[0], side="right")-1
	endBin = np.searchsorted(boundaries, endPoint[0], side="right")-1

	trackPartitions[trackFolders[startBin]].append(t)

	if startBin == endBin:
		t.save(f"{trackFolders[startBin]}/track_{t.id}.yaml")
	else:
		print("Track spans multiple bins, saving to starting bin...")
		t.save(f"{trackFolders[startBin]}/track_{t.id}.yaml")

print("Done partitioning tracks. Plotting each partition...")

# Create image view for displaying background image and tracks
imageView = OverlayView(grid=None)

# Add image background
img, timestamp = data.read()
undistortedImg = unTrans.transformImage(img)
transformedImg = pxTrans.transformImage(undistortedImg)
imageView.updateImage(transformedImg, timestamp)

for folder in trackFolders:
	transTracks = np.asarray([pxTrans.transformTrack(unTrans.transformTrack(t)) for t in trackPartitions[folder]])

	imageView.plotTracks(transTracks, labelled=True)
	imageView.setTitle(f"Tracks in {folder} bin")
	imageView.save(f"{folder}_tracks_plot.pdf")
	imageView.clearTracks()
	imageView.updateImage(transformedImg, timestamp)