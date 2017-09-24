import sys
import glob
import os

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

""" Want to process output from pipeline runs
	- Load relevant database
	- Read tracks in
	- Generate measurements from tracks
	- Run field approximations using measurements
	- Run drift analysis
	- Plot field, variance, tracks, measurement distribution
"""

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

# Load training tracks
trackDir = f"{outputDir}/tracks/good"
trackFiles = sorted(glob.glob(f"{trackDir}/track_*.yaml"))
tracks = [Track.from_file(t) for t in trackFiles]

# Unwarp tracks
undistortedTracks = [undistortTransform.transformTrack(t) for t in tracks]
transformedTracks = [pxTransform.transformTrack(t) for t in undistortedTracks]

# Load and transform background image
img, timestamp = data.read()
undistortedImg = undistortTransform.transformImage(img)
transformedImg = pxTransform.transformImage(undistortedImg)

# Initialize field approximator
gp = field_approx.gp.GPApproximator()

# Construct measurement filtering grid and view for measurement density
measurementGrid = Grid(*data.imgSize, 200,100)
measurementsPerCell = 5
measurementView = OverlayView(grid=measurementGrid)
measurementView.updateImage(transformedImg, timestamp)

# Initialize measurement filtering database
mDB = MeasurementDB(measurementGrid, measurementsPerCell)

# Extract and filter measurements
for t in transformedTracks:
	mDB.addMeasurements(t.measureVelocity(minDist=0.0, scoring='composite'))

# Run GPR
measurements = mDB.getMeasurements(measurementsPerCell=1)
if len(measurements) > 0:
	gp.clearMeasurements()
	gp.addMeasurements(measurements)
	approxField = gp.approximate()

# Plot measurement density
measurementDensityFile = f"{outputDir}/measurement_density.pdf"
measurementView.hist2d(mDB.getMeasurements())
measurementView.plot()
measurementView.save(measurementDensityFile)

# Save approximation to file
approxFieldFile = f"{outputDir}/approx.field"
approxField.save(approxFieldFile)

# Setup grid and view for displaying field
displayGrid = Grid(*data.imgSize, 60,50)
fieldView = FieldOverlayView(displayGrid)
fieldView.updateImage(transformedImg)

# Draw and save field
fieldViewFile = f"{outputDir}/approx_field.pdf"
fieldView.drawField(approxField)
fieldView.setTitle('Field Approximation')
fieldView.save(fieldViewFile)
fieldView.clearField()

# Draw and save field variances
xVarFile = f"{outputDir}/approx_field_x_var.pdf"
fieldView.drawVariance(0)
fieldView.setTitle('Field Approximation Variance (X Axis)')
fieldView.save(xVarFile)

fieldView.clearAxes()
fieldView.updateImage(transformedImg, timestamp)
yVarFile = f"{outputDir}/approx_field_y_var.pdf"
fieldView.drawVariance(1)
fieldView.setTitle('Field Approximation Variance (Y Axis)')
fieldView.save(yVarFile)


# Run Drift Analysis on track
da = DriftAnalysis(approxField)
#simTrack, diffs = da.evaluate(transformedTracks[0])
"""
driftAnalysisFile = f"{outputDir}/drift_analysis.pdf"
imageView.clearTracks()
imageView.updateImage(transformedImg, timestamp)
imageView.plotTracks([transformedTracks[0], simTrack], [(0.0,1.0,0.0), (1.0,0.0,0.0)])
imageView.plot()
imageView.save(driftAnalysisFile)
"""