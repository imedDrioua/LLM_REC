Data loader module
===================
This module define data class to load the datasets and provide some utility functions to sample users,
and describe the datasets.


..  tip::
    For each new dataset, a new class should be created, and the class should implement the following methods:

    - __init__: initialize the class with the data directory and the batch size

    - __len__: return the length of all the datasets as dictionary

    - get_dataset: return the dataset by name

    - get_all_datasets: return all the datasets

    - sample: sample n_users from the train dataset, and return the users, positive and negative books

    - describe: print the shape of all the datasets, the number of interactions in the train matrix, and the sparsity of the train matrix.


.. automodule:: data_loader
   :members:
   :undoc-members:
   :show-inheritance:
