import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

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

from lspiv_toolkit.config import PipelineConfig
from lspiv_toolkit.filtering.measurements import MeasurementDB

# Parse output folder
outputDir = os.path.abspath(sys.argv[1])

# Create output file names
trackPlotFilename = f"{outputDir}/track_plot.pdf"

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

# Load tracks
trackDir = f"{outputDir}/tracks"
testTrackFiles = sorted(glob.glob(f"{trackDir}/test/track_*.yaml"))
knownTrackFiles = sorted(glob.glob(f"{trackDir}/known/track_*.yaml"))
testTracks = [Track.from_file(t) for t in testTrackFiles]
knownTracks = [Track.from_file(t) for t in knownTrackFiles]

# Unwarp tracks
warpedTestTracks = [undistortTransform.transformTrack(t) for t in testTracks]
transformedTestTracks = [pxTransform.transformTrack(t) for t in warpedTestTracks]
warpedKnownTracks = [undistortTransform.transformTrack(t) for t in knownTracks]
transformedKnownTracks = [pxTransform.transformTrack(t) for t in warpedKnownTracks]

# Load and transform background image
img, timestamp = data.read()
undistortedImg = undistortTransform.transformImage(img)
transformedImg = pxTransform.transformImage(undistortedImg)

# Todo: Can we speed this up?
approxFieldFile = f"{outputDir}/approx.field"
approxField = VectorField.from_file(approxFieldFile)

# Run Drift analysis on known tracks
da = DriftAnalysis(approxField)

driftAnalysisFile = f"{outputDir}/drift_analysis_known.pdf"
imageView.updateImage(transformedImg, timestamp)

for tr in transformedKnownTracks:
	simTrack, x, y, errors, normErrors = da.evaluate(tr)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(tr.times[-len(normErrors):], normErrors, color='red', label='Norm Error (px/s)')
	ax.plot(tr.times, errors, label='Error (px)')
	ax.legend()
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Error and Error/s (px and px/s)")
	plt.show()
	fig.savefig(f"{outputDir}/ErrorsVsTime_track_{tr.id}.pdf")
	plt.close(fig)

	imageView.plotTracks([tr, simTrack], [(0.0,1.0,0.0), (1.0,0.0,0.0)])
	imageView.plotAnnotatedLine(x, y)
	imageView.plot()

imageView.save(driftAnalysisFile)

driftAnalysisFile = f"{outputDir}/drift_analysis_test.pdf"
imageView.clearTracks()
imageView.updateImage(transformedImg, timestamp)

boatMass = 10. #kg

for tr in transformedTestTracks:
	simTrack, x, y, errors, normErrors = da.evaluate(tr, boatMass)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(tr.times[-len(normErrors):], normErrors, color='red', label='Norm Error (px/s)')
	ax.plot(tr.times, errors, label='Error (px)')
	ax.legend()
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Error and Error/s (px and px/s)")
	plt.show()
	fig.savefig(f"{outputDir}/ErrorsVsTime_track_{tr.id}.pdf")
	plt.close(fig)

	imageView.plotTracks([tr, simTrack], [(0.0,1.0,0.0), (1.0,0.0,0.0)])
	imageView.plotAnnotatedLine(x, y)
	imageView.plot()

imageView.save(driftAnalysisFile)