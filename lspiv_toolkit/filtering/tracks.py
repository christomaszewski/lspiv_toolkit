from ..tracking import Track

class TrackDB(object):

	def __init__(self):
		# Threshold for moving tracks to historical db
		self._historicalThreshold = 0