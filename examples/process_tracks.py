import sys
import glob
import os
import numpy as np

from context import lspiv_toolkit

from primitives.track import Track
from primitives.grid import Grid

import field_toolkit.approx as field_approx
from field_toolkit.analysis.drift import DriftAnalysis


from viz_toolkit.view import OverlayView, FieldOverlayView

from cv_toolkit.data import Dataset
from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform

from lspiv_toolkit.config import PipelineConfig
from lspiv_toolkit.filtering.measurements import MeasurementDB

# Parse output folder
outputDir = os.path.abspath(sys.argv[1])

# Create output file names
trackPlotFilename = f"{outputDir}/raw_track_plot.pdf"

# Load config file
configFile = f"{outputDir}/config.yaml"
config = PipelineConfig.from_file(configFile)

# Load dataset as specified in config file
data = Dataset.from_file(config.datasetFile)

# Create image view for displaying background image and tracks
imageView = OverlayView(grid=None)

# Setup transformation objects
undistortTransform = UndistortionTransform(data.camera)
pxTransform = PixelCoordinateTransform(data.imgSize)

# Load training tracks
print("Loading track files...")
trackDir = f"{outputDir}/tracks"
trackFiles = sorted(glob.glob(f"{trackDir}/track_*.yaml"))
tracks = np.asarray([Track.from_file(t) for t in trackFiles])

# Unwarp tracks
print("Transforming Tracks...")
transformedTracks = np.asarray([pxTransform.transformTrack(undistortTransform.transformTrack(t)) for t in tracks])

# Load and transform background image
print("Plotting all tracks...")
img, timestamp = data.read()
undistortedImg = undistortTransform.transformImage(img)
transformedImg = pxTransform.transformImage(undistortedImg)

# Add image background
imageView.updateImage(transformedImg, timestamp)

# Plot tracks on image
imageView.plotDirectedTracks(transformedTracks)
imageView.save(trackPlotFilename)

# Prior Knowledge of general river direction (unit vector)
# For Llobregat datasets water flows in positive y direction
riverFlowPrior = np.array((0.0, 1.0))

# Track partition labels and thresholds
labels = ["na", "poor", "fair", "good"]
thresholds = [-1.0, -0.25, 0.25, 0.7]
partitions = dict([(l, []) for l in labels])

# Sort tracks into sets based on their agreement with prior
# Use transformed tracks to compute agreement
print("Partitioning tracks by correspondence with river flow prior...")
for index, track in enumerate(transformedTracks):
	_, endPoint = track.getLastObservation()
	_, startPoint = track.getFirstObservation()

	vec = endPoint - startPoint
	vec = vec / np.linalg.norm(vec)

	trackAgreement = np.dot(riverFlowPrior, vec)

	l = labels[np.searchsorted(thresholds, trackAgreement, side='right')-1]
	partitions[l].append(index)
print("Done partitioning.")

for label, indices in partitions.items():
	print(f"Saving {label} Tracks...")
	# If partition folders don't exist, create them
	trackPartitionDir = f"{trackDir}/{label}"
	if not os.path.exists(trackPartitionDir):
		os.makedirs(trackPartitionDir)

	# Save selected original tracks in partition folder
	for t in tracks[indices]:
		t.save(f"{trackPartitionDir}/track_{t.id}.yaml")

	print(f"Plotting {label} Tracks")
	# Plot and save partition of unwarped tracks
	imageView.clearTracks()
	imageView.updateImage(transformedImg, timestamp)
	imageView.plotDirectedTracks(transformedTracks[indices])
	imageView.save(f"{outputDir}/{label}_track_plot.pdf")
