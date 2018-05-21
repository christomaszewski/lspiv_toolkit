import sys
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from context import lspiv_toolkit

from primitives.track import Track
from primitives.grid import Grid

import field_toolkit.approx as field_approx
from field_toolkit.analysis.drift import DriftAnalysis
from field_toolkit.core.fields import VectorField

from viz_toolkit.view import OverlayView, FieldOverlayView

from cv_toolkit.data import Dataset
from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.transform.common import PixelCoordinateTransform

from lspiv_toolkit.config import PipelineConfig, ApproximationConfig
from lspiv_toolkit.filtering.measurements import MeasurementDB


# Parse output folder and generate other folder names
outputDir = os.path.abspath(sys.argv[1])
trackDir = f"{outputDir}/tracks"
testTrackDir = f"{trackDir}/test"
trainTrackDir = f"{trackDir}/train"
evalTrackDir = f"{trackDir}/eval"

# Parse field filename
fieldFile = os.path.abspath(sys.argv[2])

# Load field
field = VectorField.from_file(fieldFile)

# Load config file
configFile = f"{outputDir}/config.yaml"
config = PipelineConfig.from_file(configFile)

# Load dataset as specified in config file
data = Dataset.from_file(config.datasetFile)

# Create image view for displaying background image and tracks
imageView = OverlayView(grid=None)

# Setup transformation objects
unTrans = UndistortionTransform(data.camera)
pxTrans = PixelCoordinateTransform(data.imgSize)

# Load tracks
testTrackFiles = glob.glob(f"{testTrackDir}/track_*.yaml")
testTracks = Track.from_file_list(testTrackFiles)

# Unwarp tracks
warpedTestTracks = [unTrans.transformTrack(t) for t in testTracks]
transformedTestTracks = [pxTrans.transformTrack(t) for t in warpedTestTracks]

# Load and transform background image
img, timestamp = data.read()
undistortedImg = unTrans.transformImage(img)
transformedImg = pxTrans.transformImage(undistortedImg)

# Setup visualization of drift analysis
driftView = OverlayView(grid=None)
driftView.updateImage(transformedImg, timestamp)

# Run Drift analysis on known tracks
da = DriftAnalysis(field)

# Initialize lists to store different metrics for each track
meanErrors = []
maxErrors = []
meanNormErrors = []
maxNormErrors = []

for t in transformedTestTracks:
	print(f"Processing Test track {t.id}...")
	simTrack, x, y, errors, normErrors = da.evaluate(t, mass=0.00001)

	meanErrors.append(np.mean(errors))
	maxErrors.append(np.max(errors))
	meanNormErrors.append(np.mean(normErrors))
	maxNormErrors.append(np.max(normErrors))

	driftView.plotTracks([t,simTrack], [(0.,1.,0.), (1.,0.,0.)])
	driftView.plotAnnotatedLine(x,y)
	driftView.plot()

print(meanErrors, np.mean(meanErrors))
print(maxErrors, np.mean(maxErrors))
print(meanNormErrors, np.mean(meanNormErrors))
print(maxNormErrors, np.mean(maxNormErrors))

driftView.save(f"{outputDir}/eval_drift_analysis.pdf")