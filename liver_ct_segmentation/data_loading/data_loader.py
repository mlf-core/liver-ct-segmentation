import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

class VolumeDataset(Dataset):

    def __init__(self, ids, path="ds/", apply_trans=True):
        """
        Args:

        assumes the filenames of an image pair (input and label) are img_<ID>.pt and lab_<ID>.pt
            
        """
                        
        self.inputs = []
        self.labels = []

        for sample in ids:
            
            #print('sample: ' + str(sample))

            vol = torch.load(path + "img_" + str(sample) + ".pt")
            label = torch.load(path + "lab_" + str(sample) + ".pt")

            #############
            #change type
            #vol = vol.astype(np.float32)
            label = label.astype(np.float32)

            vol = np.expand_dims(vol, axis=0) ##add channel dim

            vol = torch.tensor(vol)
            label = torch.tensor(label)
            #label = label.type(torch.long)

            self.inputs.append(vol)
            self.labels.append(label)

        ###################################
        #transforms (e.g. RandomAffine)
        # self.apply_trans = apply_trans

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        vol = self.inputs[idx]
        label = self.labels[idx]

        pair = (vol, label)

        # ###########
        # #transform
        # if self.apply_trans:
        #     pair = self.apply_transformation(vol, label)
        # ##########

        return pair


class LitsDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of the data module with a train and test dataset, as well as a loader for each.
        The dataset is th example MNIST dataset
        """
        super(LitsDataModule, self).__init__()
        self.df_train = None
        self.df_test = None
        self.train_data_loader = None
        self.test_data_loader = None
        self.args = kwargs

        ########define train/test sets
        self.train_ids = [48, 3, 23, 75, 22, 109, 73, 130, 115, 86, 121, 67, 97, 116, 14, 125, 52, 84, 129, 58, 49, 110,
                        43, 88, 25, 4, 89, 50, 29, 94, 53, 16, 2, 46, 92, 113, 44, 15, 111, 124, 69, 47, 5, 104, 54, 37, 76,
                        119, 13, 34, 21, 103, 80, 91, 82, 35, 19, 6, 72, 59, 105, 83, 20, 128, 120, 57, 101, 30, 28, 24, 8, 41,
                        31, 95, 63, 0, 126, 11, 1, 85, 7, 33, 127, 56, 118, 70, 26, 81, 78, 40, 55, 122, 99, 71, 60, 42, 87, 9, 93,
                        108, 39, 18, 77, 90, 68, 32, 102, 79, 12, 96, 112, 36, 65, 123, 66, 10, 107, 98]
        
        self.test_ids = [17, 64, 27, 114, 74, 45, 61, 38, 106, 100, 117, 51, 62]
        ########


    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        self.df_train = VolumeDataset(self.train_ids, path=self.args['dataset_path'], apply_trans=False)
        self.df_test = VolumeDataset(self.test_ids, path=self.args['dataset_path'], apply_trans=False)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(self.df_train, batch_size=self.args['training_batch_size'], num_workers=self.args['num_workers'], shuffle=True)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'], shuffle=False)

    def val_dataloader(self):
        """
        :return: output - Val data loader for the given input
        """
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'], shuffle=False)

