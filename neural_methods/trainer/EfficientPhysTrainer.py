"""Trainer for EfficientPhys."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.EfficientPhys_FSAM import EfficientPhys_FSAM
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class EfficientPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        in_channels = config.MODEL.EFFICIENTPHYS.CHANNELS
        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            dev_list = [int(d) for d in config.DEVICE.replace("cuda:", "").split(",")]
            self.device = torch.device(dev_list[0])     #currently toolbox only supports 1 GPU
            self.num_of_gpu = 1     #config.NUM_OF_GPU_TRAIN  # set number of used GPUs
        else:
            self.device = torch.device("cpu")  # if no GPUs set device is CPU
            self.num_of_gpu = 0  # no GPUs used

        if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_train":

            if config.MODEL.NAME == "EfficientPhys":
                self.model = EfficientPhys(
                    in_channels=in_channels,
                    frame_depth=self.frame_depth,
                    img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H,
                    batch_size=self.batch_size,
                    device=self.device)
            else:
                self.model = EfficientPhys_FSAM(
                    in_channels=in_channels,
                    frame_depth=self.frame_depth,
                    img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H,
                    batch_size=self.batch_size,
                    device=self.device)

            if torch.cuda.device_count() > 0 and self.num_of_gpu > 0:  # distribute model across GPUs
                self.model = torch.nn.DataParallel(self.model, device_ids=[self.device])  # data parallel model
            else:
                self.model = torch.nn.DataParallel(self.model).to(self.device)

            self.num_train_batches = len(data_loader["train"])
            # self.criterion = torch.nn.MSELoss()
            self.criterion = Neg_Pearson()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=config.TRAIN.LR, 
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)
        
        elif config.TOOLBOX_MODE == "only_test":
            if config.MODEL.NAME == "EfficientPhys":
                self.model = EfficientPhys(
                    in_channels=in_channels,
                    frame_depth=self.frame_depth,
                    img_size=config.TEST.DATA.PREPROCESS.RESIZE.H,
                    batch_size=self.batch_size,
                    device=self.device)
            else:
                self.model = EfficientPhys_FSAM(
                    in_channels=in_channels,
                    frame_depth=self.frame_depth,
                    img_size=config.TEST.DATA.PREPROCESS.RESIZE.H,
                    batch_size=self.batch_size,
                    device=self.device)

            if torch.cuda.device_count() > 0 and self.num_of_gpu > 0:  # distribute model across GPUs
                self.model = torch.nn.DataParallel(self.model, device_ids=[self.device])  # data parallel model
            else:
                self.model = torch.nn.DataParallel(self.model).to(self.device)
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data

                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)

                data = data[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(max(self.num_of_gpu, 1), 1, 1, 1)
                data = torch.cat((data, last_frame), 0)

                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
                labels = labels.view(-1, 1)
                labels = labels[:(N * D) // self.base_len * self.base_len]

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)

                # Not to be done for MSE loss
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize

                loss = self.criterion(pred_ppg, labels)

                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                
                if len(labels_valid.shape) > 2:
                    labels_valid = labels_valid[..., 0]     # Compatibility wigth multi-signal labelled data
                labels_valid = (labels_valid - torch.mean(labels_valid)) / torch.std(labels_valid)  # normalize
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_valid[-1, :, :, :], 0).repeat(max(self.num_of_gpu, 1), 1, 1, 1)
                data_valid = torch.cat((data_valid, last_frame), 0)
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)

                # last_sample = torch.unsqueeze(labels_valid[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels_valid = torch.cat((labels_valid, last_sample), 0)
                # labels_valid = torch.diff(labels_valid, dim=0)
                # labels_valid = labels_valid / torch.std(labels_valid)  # normalize
                # labels_valid[torch.isnan(labels_valid)] = 0
                
                # Not to be done for MSE loss
                pred_ppg_valid = (pred_ppg_valid - torch.mean(pred_ppg_valid)) / torch.std(pred_ppg_valid)  # normalize                

                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(self.device), test_batch[1].to(self.device)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                
                if len(labels_test.shape) > 2:
                    labels_test = labels_test[..., 0]     # Compatibility wigth multi-signal labelled data
                labels_test = (labels_test - torch.mean(labels_test)) / torch.std(labels_test)  # normalize

                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(max(self.num_of_gpu, 1), 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)

                # last_sample = torch.unsqueeze(labels_test[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels_test = torch.cat((labels_test, last_sample), 0)
                # labels_test = torch.diff(labels_test, dim=0)
                # labels_test = labels_test / torch.std(labels_test)  # normalize
                # labels_test[torch.isnan(labels_test)] = 0

                # Not to be done for MSE loss
                pred_ppg_test = (pred_ppg_test - torch.mean(pred_ppg_test)) / torch.std(pred_ppg_test)  # normalize
                
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
