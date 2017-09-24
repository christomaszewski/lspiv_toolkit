import sys
import glob
import os
from scipy.interpolate import interp1d
import numpy as np

from context import lspiv_toolkit

from primitives import Track, Grid

import field_toolkit.approx as field_approx
from field_toolkit.core.fields import VectorField
from field_toolkit.analysis.drift import DriftAnalysis

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
print("Loading tracks...")
trackDir = f"{outputDir}/tracks/good"
trackFiles = glob.glob(f"{trackDir}/track_*.yaml")
tracks = Track.from_file_list(trackFiles)
trackDict = {t.id: t for t in tracks}

# Initialize measurement filtering database
measurementGrid = Grid(*data.imgSize, 384, 216)
measurementBinCapacity = 1000
measurementsPerCell = 1
mDB = MeasurementDB(measurementGrid, measurementBinCapacity)

# Extract and filter measurements
print("Extracting measurements...")
for t in trackDict.values():
	unwarpedTrack = pxTrans.transformTrack(unTrans.transformTrack(t))
	mDB.addMeasurements(unwarpedTrack.measureVelocity(scoring='time'))

print("Computing measurement coverage...")
measurementCoverage = mDB.getUniqueCoverage(measurementsPerCell)

# Save track coverage to folder
print("Saving tracks...")
coverageDir = f"{outputDir}/tracks/unique_coverage"
if not os.path.exists(coverageDir):
	os.makedirs(coverageDir)

for tID in measurementCoverage:
	trackDict[tID].save(f"{coverageDir}/track_{tID}.yaml")