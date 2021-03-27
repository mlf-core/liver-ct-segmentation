Usage
=============

Setup
-------

mlf-core based mlflow projects require either Conda or Docker to be installed.
The usage of Docker is highly preferred, since it ensures that system-intelligence can fetch all required and accessible hardware.
This cannot be guaranteed for MacOS let alone Windows environments.

Conda
+++++++

There is no further setup required besides having Conda installed and CUDA configured for GPU support.
mlflow will create a new environment for every run.

Docker
++++++++

If you use Docker you should not need to build the Docker container manually, since it should be available on Github Packages or another registry.
However, if you want to build it manually for e.g. development purposes, ensure that the names matches the defined name in the ``MLproject``file.
This is sufficient to train on the CPU. If you want to train using the GPU you need to have the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ installed.

Training
-----------

Training on the CPU
+++++++++++++++++++++++

Set your desired environment in the MLproject file. Start training using ``mlflow run .``.
No further parameters are required.

Training using GPUs
+++++++++++++++++++++++

Conda environments will automatically use the GPU if available.
Docker requires the accessible GPUs to be passed as runtime parameters. To train using all gpus run ``mlflow run . -A t-A gpus=all -P gpus=<<num_of_gpus>> -P acc=ddp``.
To train only on CPU it is sufficient to call ``mlflow run . -A t``. To train on a single GPU, you can call ``mlflow run . -A t -A gpus=all -P gpus=1`` and for multiple GPUs (for example 2)
``mlflow run . -A t -A gpus=all -P gpus=2 -P accelerator=ddp``.
You can replace ``all`` with specific GPU ids (e.g. 0) if desired.

Parameters
+++++++++++++++

- gpus                        Number of gpus to train with                             [2:         int]
- accelerator                 Accelerator connecting to the Lightning Trainer          ['ddp':  string]
- max_epochs:                 Number of epochs to train                                [1000:         int]
- general-seed:               Python, Random, Numpy seed                               [0:         int]
- pytorch-seed:               Pytorch specific seed                                    [0:         int]
- training-batch-size:        Batch size for training batches                          [1:        int]
- test-batch-size:            Batch size for test batches                              [1:      int]
- lr:                         Learning rate of the optimizer                           [0.0001:    float]
- log-interval:               Number of batches to train for before logging            [3000:       int]
- class-weights:              Class weights for loss function (separated by commas)    ['0.2, 1.0, 2.5':       string]
- test-percent:               Can be used to separate train and test sets (unused)     [0.15:       float]
- test-epochs:           	  Number of epochs between validations            		   [10:       int]
- dataset-path:               Path to dataset            							   ['/data/':       string]
- dataset-size:               Can be used to reduce dataset size (unused)	           [131:       int]
- n-channels:                 Number of input channels for U-Net            		   [1:       int]
- n-class:               	  Number of classes for U-Net            				   [3:       int]
- num_workers:                Number of workers for data loading					   [24:       int]
- dropout-rate:               Dropout rate for U-Net            					   [0.25:       float]
