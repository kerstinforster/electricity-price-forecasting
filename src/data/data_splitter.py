""" This class implements a data splitter that creates our training, test and
validation sets, which are ready to be used for our models"""

# TODO: implement necessary functions for data splitting in training, test
#  and validation sets

# TODO: also need functionality to create labels y_t (t+1, t+24, t+168)

# For every label y_t (n_labels x 1) there has to be an input x_t which is a
# matrix (n_features x n_time_steps)
# The labels have to created from the spot market price time series data.
# Every label y_t has to be either t+1, t+24 or t+168 timesteps in the future
# relative to its input x_t.

# TODO: find efficient solution with Jakob (saving this many matrices x_t might
#  be inefficient)
