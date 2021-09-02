from src.model_evaluator import ModelEvaluator
from src.data.dataset_generator import DatasetGenerator
import numpy as np

model_eval = ModelEvaluator()

dg = DatasetGenerator(['all'])
dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')
test_set = dataset.tail(500)
print(test_set.columns)

noise = np.random.normal(0, 20, len(test_set))  # to make 'predictions' different
test_set['PredictedSPOTPrice'] = test_set['SPOTPrice'] + noise


# in order to create the plots we have to 'repackage' our predictions back into
# a pandas dataframe together with the corresponding timestamps
# see function doc for more detailed infos

model_eval.create_all_comparison_plots(
    y_pred=test_set,
    y_true=test_set,
    save_at='pred_comparison_plots/',
    n_steps=72,
    show_plots=True
)
