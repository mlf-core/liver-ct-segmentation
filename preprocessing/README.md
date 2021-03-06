# Liver CT Segmentation Data Preprocessing

This Docker container and the corresponding jupyter notebook reproduces the data preprocessing to train models.

# Data

Obtain the training data from https://competitions.codalab.org/competitions/17094 by signing up, registering for the competition (never ends) and downloading the data.

# Building

1. Ensure that the `start.sh` file has executable permissions.
2. Build the container: `docker build -t mlf-core/liver_ct_seq:latest .`

# Usage

1. Start the container `docker run -it -v /path/to/full_dataset:/home/jovyan/full_dataset -p 8888:8888 mlf-core/liver_ct_seq:latest /bin/bash`
2. Run bash start.sh
3. Run the cells inside `liver_ct_seq_dataset.ipynb`
4. The training data is found inside `unet/ds`

# Tipps

To switch to the root user run the container with `-e GRANT_SUDO=yes --user root`
