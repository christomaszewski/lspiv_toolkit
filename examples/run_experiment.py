import sys
import glob
import os
import time
import matplotlib.pyplot as plt

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


"""
	Setup all relevant paths and load config files
"""
print("Reading config files and setting up file paths...")

# Parse and load approximation config 
if len(sys.argv) < 2:
	# If no config supplied, load default
	approxConfigFile = "approx_config.yaml"
else:
	approxConfigFile = os.path.abspath(sys.argv[1])

approxConfig = ApproximationConfig.from_file(approxConfigFile)

# Extract relevant folder names from approximation config
outputDir = approxConfig.outputDir
trackDir = approxConfig.trackDir
resultsDir = f"{outputDir}/results_{time.strftime('%d_%m_%Y_%H_%M_%S')}"
trainingResultsDir = f"{resultsDir}/training"
evalResultsDir = f"{resultsDir}/eval"

# Construct paths for track subsets
trainingDataDir = f"{trackDir}/train"
testDataDir = f"{trackDir}/test"
evalDataDir = f"{trackDir}/eval"

# Load pipeline config
configFile = f"{outputDir}/config.yaml"
config = PipelineConfig.from_file(configFile)


"""
	Initialize all necessary components for track processing and field
	reconstruction and analysis
"""
print("Intializing necessary processing, approximation, and analysis components...")

# Load image dataset from disk
imgDB = Dataset.from_file(config.datasetFile)

# Setup transformation objects
unTrans = UndistortionTransform(imgDB.camera)
pxTrans = PixelCoordinateTransform(imgDB.imgSize)

# Initialize specified field approximator
if approxConfig.approximationMethod == 'simple':
	gp = field_approx.gp.GPApproximator()
elif approxConfig.approximationMethod == 'coregionalized':
	gp = field_approx.gp.CoregionalizedGPApproximator()
elif approxConfig.approximationMethod == 'sparse':
	gp = field_approx.gp.SparseGPApproximator()
elif approxConfig.approximationMethod == 'integral':
	gp = field_approx.gp.IntegralGPApproximator()
else:
	print("Error: Unknown approximation method")
	exit()

# Construct measurement filtering grid
measurementGrid = Grid(*imgDB.imgSize, *approxConfig.measurementGridDim)
measurementsPerCell = approxConfig.measurementsPerCell

# Initialize measurement filtering database
mDB = MeasurementDB(measurementGrid, measurementsPerCell)

# Initialize drift analysis tool
da = DriftAnalysis()


"""
	Load data from disk and process for usage
"""
print("Loading data from disk...")

# Load first image from dataset to use as background image
print("Images...")
img, timestamp = imgDB.read()

# Undistort image and transform for plotting
warpImg = unTrans.transformImage(img)
transImg = pxTrans.transformImage(warpImg)

# Load tracks from disk (may take some time if there are alot)
print("Training Track Data...")
trainingFiles = glob.glob(f"{trainingDataDir}/track_*.yaml")
trainTracks = Track.from_file_list(trainingFiles)
print("Test Track Data...")
testFiles = glob.glob(f"{testDataDir}/track_*.yaml")
testTracks = Track.from_file_list(testFiles)
print("Evaluation Track Data...")
evalFiles = glob.glob(f"{evalDataDir}/track_*.yaml")
evalTracks = Track.from_file_list(evalFiles)

# Unwarp tracks for processing and plotting
trainTracksWarped = [pxTrans.transformTrack(unTrans.transformTrack(t)) for t in trainTracks]
testTracksWarped = [pxTrans.transformTrack(unTrans.transformTrack(t))  for t in testTracks]
evalTracksWarped = [pxTrans.transformTrack(unTrans.transformTrack(t))  for t in evalTracks]


"""
	Ready to start processing data and generating results. Create results 
	directory and start processing.
"""
print("Starting experiment...")

# Create results directories if not present
if not os.path.exists(resultsDir):
	os.makedirs(resultsDir)

if not os.path.exists(trainingResultsDir):
	os.makedirs(trainingResultsDir)

if not os.path.exists(evalResultsDir):
	os.makedirs(evalResultsDir)


# Save approx config to results directory
approxConfig.save(f"{resultsDir}/approx_config.yaml")

# Extract and filter measurements
print("Extracting training measurements")
for t in trainTracksWarped:
	mDB.addMeasurements(t.measureVelocity(scoring=approxConfig.scoringMethod))

trainMeasurements = mDB.getMeasurements()

# Run GPR
if len(trainMeasurements) > 0:
	gp.clearMeasurements()
	gp.addMeasurements(trainMeasurements)
	approxField = gp.approximate()

# Save field approximation
approxField.save(f"{trainingResultsDir}/approx.field")

# Prepare objects for plotting
measurementView = OverlayView(grid=measurementGrid)
measurementView.updateImage(transImg, timestamp)
displayGrid = Grid(*imgDB.imgSize, 64,36)
varGrid = Grid(*imgDB.imgSize, 120, 100)
fieldView = FieldOverlayView(displayGrid)
fieldView.updateImage(transImg, timestamp)
driftView = OverlayView(grid=None)
driftView.updateImage(transImg, timestamp)

# Plot and save visuals
measurementView.hist2d(trainMeasurements)
measurementView.setTitle("Training Measurement Density")
measurementView.save(f"{trainingResultsDir}/training_measurement_density.pdf")

fieldView.drawField(approxField)
fieldView.setTitle("Vector Field Approximation using LSPIV Data")
fieldView.save(f"{trainingResultsDir}/field_approx.pdf")
fieldView.clearAxes()

fieldView.updateImage(transImg, timestamp)
fieldView.drawVariance(0, varGrid)
fieldView.setTitle("X Component Variance for LSPIV Approximation")
fieldView.save(f"{trainingResultsDir}/x_field_variance.pdf")
fieldView.clearAxes()

fieldView.updateImage(transImg, timestamp)
fieldView.drawVariance(1, varGrid)
fieldView.setTitle("Y Component Variance for LSPIV Approximation")
fieldView.save(f"{trainingResultsDir}/y_field_variance.pdf")

# Prepare figure for drift analysis error plots
fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)

# Run drift analysis with test data
print("Running drift analysis with test data...")
da.updateField(approxField)

for t in testTracksWarped:
	print(f"Processing test track {t.id}...")
	simTrack, x, y, errors, normErrors = da.evaluate(t, mass=0.00001)

	ax.plot(t.times[-len(normErrors):], normErrors, color='red', label='Normalized Error')
	ax.plot(t.times, errors, label='Error')
	ax.legend()
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Error and Normalized Error (px)")
	ax.set_title("Error in Track Prediction vs. Time")
	plt.show()
	plt.pause(0.001)
	fig.savefig(f"{trainingResultsDir}/drift_errors_{t.id}.pdf", dpi=300, bbox_inches='tight')
	ax.clear()

	driftView.plotTracks([t, simTrack], [(0.,1.,0.), (1.,0.,0.)])
	driftView.plotAnnotatedLine(x, y)
	driftView.plot()

driftView.setTitle("Drift Prediction Errors using LSPIV Approximation")
driftView.save(f"{trainingResultsDir}/drift_analysis.pdf")


"""
	Done with first pass of generating results. Now add eval tracks to 
	training data and repeat
"""
print("Processing tracks marked for evaluation...")

# Extract and filter measurements
print("Extracting evaluation measurements")
for t in evalTracksWarped:
	mDB.addMeasurements(t.measureVelocity(scoring=approxConfig.scoringMethod))

augmentedMeasurements = mDB.getMeasurements()

# Run GPR
if len(augmentedMeasurements) > 0:
	gp.clearMeasurements()
	gp.addMeasurements(augmentedMeasurements)
	approxField = gp.approximate()

# Save field approximation
approxField.save(f"{evalResultsDir}/approx.field")

# Clear and prepare visual objects
measurementView.clearTracks()
measurementView.updateImage(transImg, timestamp)
driftView.clearTracks()
driftView.updateImage(transImg, timestamp)
fieldView.clearAxes()
fieldView.updateImage(transImg, timestamp)

# Plot and save visuals
measurementView.hist2d(trainMeasurements)
measurementView.setTitle("Evaluation Measurement Density")
measurementView.save(f"{evalResultsDir}/evaluation_measurement_density.pdf")

fieldView.drawField(approxField)
fieldView.setTitle("Vector Field Approximation using Augmented LSPIV Data")
fieldView.save(f"{evalResultsDir}/field_approx.pdf")
fieldView.clearAxes()

fieldView.updateImage(transImg, timestamp)
fieldView.drawVariance(0, varGrid)
fieldView.setTitle("X Component Variance for Augmented LSPIV Approximation")
fieldView.save(f"{evalResultsDir}/x_field_variance.pdf")
fieldView.clearAxes()

fieldView.updateImage(transImg, timestamp)
fieldView.drawVariance(1, varGrid)
fieldView.setTitle("Y Component Variance for Augmented LSPIV Approximation")
fieldView.save(f"{evalResultsDir}/y_field_variance.pdf")

# Run drift analysis with test data
print("Running drift analysis with test data...")
da.updateField(approxField)

for t in testTracksWarped:
	simTrack, x, y, errors, normErrors = da.evaluate(t, mass=0.00001)

	ax.plot(t.times[-len(normErrors):], normErrors, color='red', label='Normalized Error')
	ax.plot(t.times, errors, label='Error')
	ax.legend()
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Error and Normalized Error (px)")
	ax.set_title("Error in Track Prediction vs. Time")
	plt.show()
	plt.pause(0.001)
	fig.savefig(f"{evalResultsDir}/drift_errors_{t.id}.pdf", dpi=300, bbox_inches='tight')
	ax.clear()

	driftView.plotTracks([t, simTrack], [(0.,1.,0.), (1.,0.,0.)])
	driftView.plotAnnotatedLine(x, y)
	driftView.plot()

driftView.setTitle("Drift Prediction Errors using Augmented LSPIV Approximation")
driftView.save(f"{evalResultsDir}/drift_analysis.pdf")
