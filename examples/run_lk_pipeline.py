from context import lspiv_toolkit

from lspiv_toolkit.pipeline import SlimPipeline
from lspiv_toolkit.config import PipelineConfig

if __name__ == '__main__':
	
	config = PipelineConfig.from_file('config.yaml')

	pipeline = SlimPipeline(config)

	pipeline.initialize()
	pipeline.run()
	pipeline.saveTracks()

	config.save(f"{pipeline.runDir}/config.yaml")
