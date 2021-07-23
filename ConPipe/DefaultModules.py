from ConPipe.modules import DataSplit, ModelEvaluation, ModelSelection
import sklearn

DEFAULT_MODULES = {
    'ConPipe.DataSplit': DataSplit,
    'ConPipe.ModelEvaluation': ModelEvaluation,
    'ConPipe.ModelSelection': ModelSelection,
    'sklearn': sklearn
}