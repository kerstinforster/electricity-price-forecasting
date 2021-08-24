from src._modellab.model_evaluation import ModelEvaluator
from src.models.model_factory import ModelFactory

# we run one gridsearch over all our different models for each time horizon
prediction_time_horizon = 1  # t+1, t+24, or t+168

# define a parameter grid for each model
sarimax_param_grid = {
    "model_name": "sarimax",
    "n_look_back": [12, 24, 72, 168],  #
    "param_2": [1, 2, 3, 4, 5],
    "param_3": [1, 2, 3, 4, 5]
}

prophet_param_grid = {
    "model_name": "prophet"
}

lstm_param_grid = {
    "model_name": "lstm"
}


# list of dicts containing the parameter grid for the every model
all_model_param_grids = [sarimax_param_grid,
                         prophet_param_grid,
                         lstm_param_grid]


### CREATE TRAIN AND TEST SET ###
dg = DatasetGenerator(['all'])

# Get a dataset
dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')

train, test = train_test_split(dataset, 0.1)

# Create a data transformer for scaling data
dt = DataTransformer()
train = dt.fit_transform(train)
test = dt.transform_data(test)

# Create instance od ModelEvaluator for Model Ranking after training
model_eval = ModelEvaluator()


for model_param_grid in all_model_param_grids:

    # TODO: generate single model configs from param grids
    model_configs =

    for single_config in model_configs:

        ### CREATE TRAIN AND TEST SET ACCORDING TO SPECIFIED n_look_back ###
        splitter = DataSplitter(single_config["n_look_back"])  # n_look_back
        n_gap = prediction_time_horizon - 1
        x_train, y_train = splitter.split(train, n_gap)  # Gap 0, 23, 167 here
        x_test, y_test = splitter.split(test, n_gap)

        # Create and train the model with the specified config
        model = ModelFactory.get(model_name=single_config["model_name"],
                                 model_params=single_config)
        model.train(x_train, y_train, single_config)

        # Evaluate the trained model
        y_pred = model.predict(x_input=x_test)
        model_scores = model_eval.evaluate(y_pred, y_test, eval_metrics='all')

        # TODO: create unique name from config for saving
        model.save()  # saves model with current config in 'gs_trained_models'
        save_model_ranking()  # saves model ranking as txt, pkl and also prints

print_best_model_names()
# at this point we could also retrain the best model/s on the training+test set if we still want to