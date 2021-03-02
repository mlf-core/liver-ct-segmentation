from torch.autograd import Variable
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from torch.nn import functional as F
from model.unet_3d_models import UNet3D
from losses.focal_loss import FocalLoss
from metrics.metrics import iou_fnc, accuracy
import numpy as np


class LitsSegmentator(pl.LightningModule):
    def __init__(self, len_test_set: int, **kwargs):
        """
        Initializes the network
        """
        super(LitsSegmentator, self).__init__()

        self.args = kwargs

        self.optimizer = None

        self.model = UNet3D(self.args['n_channels'], self.args['n_class'], dropout_val=self.args['dropout_rate'])

        class_weights = np.array([float(i) for i in self.args['class_weights'].split(',')])
        #self.criterion = FocalLoss(apply_nonlin=None, alpha=class_weights, gamma=2)
        
        #self.len_test_set = len_test_set
        
        #self.train_acc = pl.metrics.Accuracy()
        #self.test_acc = pl.metrics.Accuracy()

        self._to_console = True


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num-workers', type=int, default=3, metavar='N', help='number of workers (default: 3)')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
        parser.add_argument('--training-batch-size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=8, help='Input batch size for testing')
        parser.add_argument('--class-weights', type=str, default='1.0,1.0,1.0', help='string of class weights (e.g. 1.0,0.5,0.2)')
        parser.add_argument('--test-percent', type=float, default=0.15, help='dataset percent for testing')
        parser.add_argument('--test-epochs', type=int, default=10, help='epochs before testing')
        parser.add_argument('--dataset-path', type=str, default='data/', help='path to dataset')
        parser.add_argument('--dataset-size', type=int, default=131, help='dataset size')
        parser.add_argument('--n-channels', type=int, default=1, help='number of input channels')
        parser.add_argument('--n-class', type=int, default=3, help='number of classes')
        parser.add_argument('--dropout-rate', type=float, default=0.25, help='dropout rate')

        return parser

    def forward(self, x):
        """
        :param x: Input data

        :return: output - mnist digit label for the input image
        """
        x = self.model(x)
        output = F.softmax(x, dim=1)

        return output

    # def cross_entropy_loss(self, logits, labels):
    #     """
    #     Initializes the loss function

    #     :return: output - Initialized cross entropy loss function
    #     """
    #     return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """

        output = {}

        x, y = train_batch
        prob_mask = self.forward(x)
        loss = self.criterion(prob_mask, y.type(torch.long))

        #loss = self.cross_entropy_loss(logits, y)
        #self.train_acc(logits, y)
        #self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        acc = accuracy(torch.argmax(prob_mask, dim=1).float(), y)
        output['acc'] = acc

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['n_class'])
        for i in range(self.args['n_class']):
            output['iou_' + str(i)] = iter_iou[i]
            output['iou_cnt_' + str(i)] = iter_count[i]

        output['loss'] = loss

        return output #{'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        """
        On each training epoch end, log the average training loss
        """

        train_avg_acc = torch.stack([train_output['acc'] for train_output in training_step_outputs]).mean()
        train_avg_loss = torch.stack([train_output['loss'] for train_output in training_step_outputs]).mean()

        train_iou_sum = torch.zeros(self.args['n_class'])
        train_iou_cnt_sum = torch.zeros(self.args['n_class'])
        for i in range(self.args['n_class']):
            train_iou_sum[i] = torch.stack([train_output['iou_' + str(i)] for train_output in training_step_outputs]).sum()
            train_iou_cnt_sum[i] = torch.stack([train_output['iou_cnt_' + str(i)] for train_output in training_step_outputs]).sum()
        iou_scores = train_iou_sum / (train_iou_cnt_sum + 1e-10)
        #iou_mean = torch.nanmedian(iou_scores)
        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        # self.log('train_avg_loss', train_avg_loss, sync_dist=True)
        # self.log('train_avg_acc', train_avg_acc, sync_dist=True)
        # self.log('train_mean_iou', iou_mean, sync_dist=True)
        # for c in range(self.args['n_class']):
        #     if train_iou_cnt_sum[c] == 0.0:
        #         iou_scores[c] = 0
        #     self.log('train_iou_' + str(c), iou_scores[c].item(), sync_dist=True)

        # if self._to_console:
        #     print('epoch {0:.1f} - loss: {1:.15f} - acc: {2:.15f} - meanIoU: {3:.15f}'.format(self.current_epoch, train_avg_loss, train_avg_acc, iou_mean))
        #     for c in range(self.args['n_class']):
        #         print('class {} IoU: {}'.format(c, iou_scores[c].item()))


    def test_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current accuracy of the model.

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """

        output = {}

        x, y = test_batch
        prob_mask = self.forward(x)
        loss = self.criterion(prob_mask, y.type(torch.long))

        #loss = self.cross_entropy_loss(logits, y)
        #self.train_acc(logits, y)
        #self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        acc = accuracy(torch.argmax(prob_mask, dim=1).float(), y)
        output['test_acc'] = acc

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['n_class'])
        for i in range(self.args['n_class']):
            output['test_iou_' + str(i)] = iter_iou[i]
            output['test_iou_cnt_' + str(i)] = iter_count[i]

        output['test_loss'] = loss

        return output

        # x, y = test_batch
        # output = self.forward(x)
        # _, y_hat = torch.max(output, dim=1)
        # self.test_acc(y_hat, y)
        # self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        # # sum up batch loss
        # data, target = Variable(x), Variable(y)  # noqa: F841
        # test_loss = F.nll_loss(output, target, reduction='sum').data.item()
        # # get the index of the max log-probability
        # pred = output.data.max(1)[1]
        # correct = pred.eq(target.data).sum()
        # return #{'test_loss': test_loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """

        test_avg_acc = torch.stack([test_output['test_acc'] for test_output in outputs]).mean()
        test_avg_loss = torch.stack([test_output['test_loss'] for test_output in outputs]).mean()

        test_iou_sum = torch.zeros(self.args['n_class'])
        test_iou_cnt_sum = torch.zeros(self.args['n_class'])
        for i in range(self.args['n_class']):
            test_iou_sum[i] = torch.stack([test_output['test_iou_' + str(i)] for test_output in outputs]).sum()
            test_iou_cnt_sum[i] = torch.stack([test_output['test_iou_cnt_' + str(i)] for test_output in outputs]).sum()
        iou_scores = test_iou_sum / (test_iou_cnt_sum + 1e-10)
        #iou_mean = torch.nanmedian(iou_scores)
        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        # self.log('test_avg_loss', test_avg_loss, sync_dist=True)
        # self.log('test_avg_acc', test_avg_acc, sync_dist=True)
        # self.log('test_mean_iou', iou_mean, sync_dist=True)
        # for c in range(self.args['n_class']):
        #     if test_iou_cnt_sum[c] == 0.0:
        #         iou_scores[c] = 0
        #     self.log('test_iou_' + str(c), iou_scores[c].item(), sync_dist=True)

        # if self._to_console:
        #     print('eval ' + str(self.current_epoch) + ' ..................................................')
        #     print('eLoss: {0:.15f} - eAcc: {1:.15f} - eMeanIoU: {2:.15f}'.format(test_avg_loss, test_avg_acc, iou_mean))
        #     for c in range(self.args['n_class']):
        #         print('class {} IoU: {}'.format(c, iou_scores[c].item()))

        ###########


        # avg_test_loss = sum([test_output['test_loss'] for test_output in outputs]) / self.len_test_set
        # test_correct = float(sum([test_output['correct'] for test_output in outputs]))
        # self.log('avg_test_loss', avg_test_loss, sync_dist=True)
        # self.log('test_correct', test_correct, sync_dist=True)

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, verbose=True,
            ),
            'monitor': 'train_avg_loss',
        }
        return [self.optimizer], [self.scheduler]
