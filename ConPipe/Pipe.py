import yaml
from ConPipe.MLModelSelection import MLModelSelection

# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


class MLDataProcess():

    def __init__(self, config):
        self.config = config


class MLDataProcess():

    def __init__(self, config):
        self.config = config


class MLFeatureExtraction():

    def __init__(self, config):
        self.config = config


class MLModelOptimization():

    def __init__(self, config):
        self.config = config


class MLResults():

    def __init__(self, config):
        self.config = config


class MLPipe():

    def __init__(self, config):
        self.config = config
        self.data_preprocess = MLDataProcess(self.config['data_preprocess'])
        self.feature_extraction = MLFeatureExtraction(self.config['feature_extraction'])
        self.data_augmentation = MLDataAugmentation(self.config['data_augmentation'])
        self.evaluation_schema = MLEvaluationSchema(self.config['evaluation_schema'])
        self.model_selection = MLModelSelection(
            self.config['model_selection'],
            verbose=self.config['general']['verbose']
        )
        self.model_evaluation = MLModelEvaluation(self.config['model_evaluation'])

    def run()