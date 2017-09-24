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

# Construct paths for track subsets
trainingDataDir = f"{trackDir}/coverage"

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
measurementBinCapacity = approxConfig.measurementBinCapacity
measurementsPerCell = approxConfig.measurementsPerCell

# Initialize measurement filtering database
mDB = MeasurementDB(measurementGrid, measurementBinCapacity)


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

# Unwarp tracks for processing and plotting
trainTracksWarped = pxTrans.transformTracks(unTrans.transformTracks(trainTracks))


"""
	Ready to start processing data and generating results. Create results 
	directory and start processing.
"""
print("Starting Approximation...")

# Create results directories if not present
if not os.path.exists(resultsDir):
	os.makedirs(resultsDir)

if not os.path.exists(trainingResultsDir):
	os.makedirs(trainingResultsDir)

# Save approx config to results directory
approxConfig.save(f"{resultsDir}/approx_config.yaml")

# Extract and filter measurements
print("Extracting training measurements")
for t in trainTracksWarped:
	mDB.addMeasurements(t.measureVelocity(scoring=approxConfig.scoringMethod))

trainMeasurements = mDB.getMeasurements(measurementsPerCell)

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
