import os
import random
from argparse import ArgumentParser
from functools import partial


import h5pickle
import numpy as np
import pytorch_lightning as pl
import torch
import tensorflow as tf
from PIL import Image, ImageFilter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from gaze_estimation_models_pytorch import GazeEstimationModelResnet18, GazeEstimationModelVGG, GazeEstimationModelPreactResnet
from rtgene_dataset import RTGENEH5Dataset
from utils.GazeAngleAccuracy import GazeAngleAccuracy
from utils.PinballLoss import PinballLoss


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects):
        super(TrainRTGENE, self).__init__()
        _loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum"),
            "pinball": partial(PinballLoss, reduction="sum")
        }
        _param_num = {
            "mse": 2,
            "pinball": 3
        }
        _models = {
            "vgg": partial(GazeEstimationModelVGG, num_out=_param_num.get(hparams.loss_fn)),
            "resnet18": partial(GazeEstimationModelResnet18, num_out=_param_num.get(hparams.loss_fn)),
            "preactresnet": partial(GazeEstimationModelPreactResnet, num_out=_param_num.get(hparams.loss_fn))
        }
        self._model = _models.get(hparams.model_base)()
        self._criterion = _loss_fn.get(hparams.loss_fn)()
        self._angle_acc = GazeAngleAccuracy()
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self.hparams = hparams

    def forward(self, left_patch, right_patch, head_pose):
        return self._model(left_patch, right_patch, head_pose)

    def training_step(self, batch, batch_idx):
        FLIP =False
        SHIFT = False

        _left_patch, _right_patch, _headpose_label, _gaze_labels = batch

        # print("=========================training step========================")
        # print("left patch shape", np.shape(_left_patch))
        # print("left patch", _left_patch)
        # print("===============================================================")
        # print("right patch shape", np.shape(_right_patch))
        # print("right patch", _right_patch)
        # print("===============================================================")
        # print("head label shape", np.shape(_headpose_label))
        # print("head pose label", _headpose_label)
        # print("===============================================================")
        # print("gaze label shape", np.shape(_gaze_labels))
        # print("gaze label", _gaze_labels)
        # print("===============================================================")

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        loss = self._criterion(angular_out, _gaze_labels)

        if FLIP:
            patch_num = len(_left_patch)

            #patch flip
            _left_patch = _left_patch.cpu().numpy()
            _right_patch = _right_patch.cpu().numpy()
            for i in range(patch_num):
                #rgb
                for j in range(3):
                    _left_patch[i][j] = np.fliplr(_left_patch[i][j])
                    _right_patch[i][j] = np.fliplr(_right_patch[i][j])

            _left_patch = torch.from_numpy(_left_patch).cuda()
            _right_patch = torch.from_numpy(_right_patch).cuda()

            # label flip
            flip_headpose_label = _headpose_label.cpu().numpy()
            flip_gaze_labels = _gaze_labels.cpu().numpy()
            for i in range(patch_num):
                flip_headpose_label[i][0] *= -1
                flip_gaze_labels[i][0] *= -1
            flip_headpose_label = torch.from_numpy(flip_headpose_label).cuda()
            flip_gaze_labels = torch.from_numpy(flip_gaze_labels).cuda()


            flip_angular_out = self.forward(_left_patch, _right_patch, flip_headpose_label)

            # angular flip
            #flip_angular_out = flip_angular_out.cpu().numpy()
            #angular_out = angular_out.cpu().numpy()

            average_angular_out = torch.zeros((patch_num, 2), requires_grad=True).cuda()
            for i in range(len(flip_angular_out)):

                average_angular_out[i][0] = (angular_out[i][0] - flip_angular_out[i][0])/2
                average_angular_out[i][1] = (angular_out[i][1] + flip_angular_out[i][1]) /2


            # flip_angular_out = torch.from_numpy(flip_angular_out).cuda()
            #angular_out =torch.from_numpy(angular_out).cuda()

            average_loss = self._criterion(average_angular_out, _gaze_labels)

            loss += average_loss


        if SHIFT:
            import cv2

            patch_num = len(_left_patch)
            _left_patch = _left_patch.cpu().numpy()
            _right_patch = _right_patch.cpu().numpy()

            # u_left_patch =np.zeros((patch_num,3,224,224))
            # u_right_patch=np.zeros((patch_num,3,224,224))
            # d_left_patch =np.zeros((patch_num,3,224,224))
            # d_right_patch = np.zeros((patch_num,3,224,224))
            l_left_patch=np.zeros((patch_num,3,224,224))
            l_right_patch=np.zeros((patch_num,3,224,224))
            r_left_patch=np.zeros((patch_num,3,224,224))
            r_right_patch = np.zeros((patch_num,3,224,224))

            #up, down, left, rigth 2px
            #shift = np.float32([[[1, 0, 0], [0, 1, -2]],[[1, 0, 0], [0, 1, 2]],[[1, 0, -2], [0, 1, 0]],[[1, 0, 2], [0, 1, 0]]])
            #up, down, left, rigth 1px
            shift = np.float32([[[1, 0, 0], [0, 1, -1]],[[1, 0, 0], [0, 1, 1]],[[1, 0, -1], [0, 1, 0]],[[1, 0, 1], [0, 1, 0]]])


            for i in range(patch_num):
                for j in range(3):
                    # u_left_patch[i][j] = cv2.warpAffine(_left_patch[i][j], shift[0],(224,224))
                    # u_right_patch[i][j] =cv2.warpAffine(_left_patch[i][j], shift[0], (224, 224))
                    #
                    # d_left_patch[i][j] = cv2.warpAffine(_left_patch[i][j], shift[1], (224, 224))
                    # d_right_patch[i][j] =cv2.warpAffine(_left_patch[i][j], shift[1], (224, 224))

                    l_left_patch[i][j] =cv2.warpAffine(_left_patch[i][j], shift[2], (224, 224))
                    l_right_patch[i][j] =cv2.warpAffine(_left_patch[i][j], shift[2], (224, 224))

                    r_left_patch[i][j] =cv2.warpAffine(_left_patch[i][j], shift[3], (224, 224))
                    r_right_patch[i][j] =cv2.warpAffine(_left_patch[i][j], shift[3], (224, 224))


            # u_left_patch = torch.from_numpy(u_left_patch).float().cuda()
            # u_right_patch = torch.from_numpy(u_right_patch).float().cuda()
            # d_left_patch = torch.from_numpy(d_left_patch).float().cuda()
            # d_right_patch = torch.from_numpy(d_right_patch).float().cuda()
            l_left_patch = torch.from_numpy(l_left_patch).float().cuda()
            l_right_patch = torch.from_numpy(l_right_patch).float().cuda()
            r_left_patch = torch.from_numpy(r_left_patch).float().cuda()
            r_right_patch = torch.from_numpy(r_right_patch).float().cuda()

            # print("====================SHIFT ============================")
            # print("R_SHIFT",np.shape(r_left_patch),"\n",r_left_patch)
            # print("L_SHIFT",np.shape(l_left_patch),"\n",l_left_patch)
            # print("U_SHIFT",np.shape(u_left_patch),"\n",u_left_patch)
            # print("D_SHIFT",np.shape(d_left_patch),"\n",d_left_patch)


            # u_angular_out = self.forward(u_left_patch, u_right_patch, _headpose_label)
            # d_angular_out = self.forward(d_left_patch, d_right_patch, _headpose_label)
            l_angular_out = self.forward(l_left_patch, l_right_patch, _headpose_label)
            r_angular_out = self.forward(r_left_patch, r_right_patch, _headpose_label)


            # shift_angular_out= torch.zeros((patch_num,2),requires_grad=True).cuda()
            #
            # for i in range(patch_num):
            #     shift_angular_out[i][0] = (angular_out[i][0]+u_angular_out[i][0] + d_angular_out[i][0] +l_angular_out[i][0]+r_angular_out[i][0])/5
            #     shift_angular_out[i][1] = (angular_out[i][1] + u_angular_out[i][1] + d_angular_out[i][1] + l_angular_out[i][1] + r_angular_out[i][1]) / 5


            r_shift_loss = self._criterion(r_angular_out, _gaze_labels)
            l_shift_loss = self._criterion(l_angular_out, _gaze_labels)

            loss =loss+ r_shift_loss +l_shift_loss


        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels = batch

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        loss = self._criterion(angular_out, _gaze_labels)
        angle_acc = self._angle_acc(angular_out[:, :2], _gaze_labels)

        return {'val_loss': loss, "angle_acc": angle_acc}

    def validation_end(self, outputs):
        _losses = torch.stack([x['val_loss'] for x in outputs])
        _angles = np.array([x['angle_acc'] for x in outputs])
        tensorboard_logs = {'val_loss': _losses.mean(), 'val_angle': np.mean(_angles)}
        return {'val_loss': _losses.mean(), 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels = batch

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        angle_acc = self._angle_acc(angular_out[:, :2], _gaze_labels)

        return {'angle_acc': angle_acc}

    def test_end(self, outputs):
        _angles = np.array([x['angle_acc'] for x in outputs])
        _mean = np.mean(_angles)
        _std = np.std(_angles)
        results = {
            'log': {'test_angle_acc': _mean, 'test_angle_std': _std}
        }
        return results

    def configure_optimizers(self):
        _params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                _params_to_update.append(param)

        _learning_rate = self.hparams.learning_rate
        # betas  0.9, 0.95 에서 논문기준인 0.9 , 0.99로 변경함. gamma 0.5 ->0.1
        _optimizer = torch.optim.Adam(_params_to_update, lr=_learning_rate, betas=(0.9, 0.99))
        _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=30, gamma=0.5)

        return [_optimizer], [_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--augment', action="store_true", dest="augment")
        parser.add_argument('--no_augment', action="store_false", dest="augment")
        parser.add_argument('--loss_fn', choices=["mse", "pinball"], default="mse")
        parser.add_argument('--batch_size', default=24, type=int)
        parser.add_argument('--batch_norm', default=True, type=bool)
        parser.add_argument('--learning_rate', type=float, default=0.00005)
        parser.add_argument('--model_base', choices=["vgg", "resnet18", "preactresnet"], default="vgg")
        return parser

    def gaussianBlur1(self,x):
            if np.random.random_sample() > 0.08:
                return x
            else:
                return x.filter(ImageFilter.GaussianBlur(radius=1))
    def gaussianBlur3(self, x):
            if np.random.random_sample() > 0.08:
                return x
            else:
                return x.filter(ImageFilter.GaussianBlur(radius=3))

    def train_dataloader(self):
        _train_transforms: None = None

        if self.hparams.augment:
            _train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
                                                    transforms.RandomGrayscale(p=0.08),
                                                    # lambda x: x if np.random.random_sample() > 0.08 else x.filter(ImageFilter.GaussianBlur(radius=1)),
                                                    # lambda x: x if np.random.random_sample() > 0.08 else x.filter(ImageFilter.GaussianBlur(radius=3)),
                                                    self.gaussianBlur1,
                                                    self.gaussianBlur3,
                                                    transforms.Resize((224, 224), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _data_train = RTGENEH5Dataset(h5_file=h5pickle.File(self.hparams.hdf5_file, mode="r"),
                                      subject_list=self._train_subjects,
                                      transform=_train_transforms)
        return DataLoader(_data_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=False)

    def val_dataloader(self):
        _data_validate = RTGENEH5Dataset(h5_file=h5pickle.File(self.hparams.hdf5_file, mode="r"), subject_list=self._validate_subjects)
        return DataLoader(_data_validate, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=False)

    def test_dataloader(self):
        _data_test = RTGENEH5Dataset(h5_file=h5pickle.File(self.hparams.hdf5_file, mode="r"), subject_list=self._test_subjects)
        return DataLoader(_data_test, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=False)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    root_dir = os.path.dirname(os.path.realpath(__file__))

    _root_parser = ArgumentParser(add_help=False)
    _root_parser.add_argument('--gpu', type=int, default=1, help='gpu to use, can be repeated for mutiple gpus i.e. --gpu 1 --gpu 2', action="append")
    _root_parser.add_argument('--hdf5_file', type=str, default=os.path.abspath(os.path.join(root_dir, "../rt_gene_dataset/rtgene_dataset_LR.hdf5")))
    _root_parser.add_argument('--dataset', type=str, choices=["rt_gene", "other"], default="rt_gene")
    _root_parser.add_argument('--save_dir', type=str, default=os.path.abspath(os.path.join(root_dir, '../model_nets/pytorch_checkpoints')))
    _root_parser.add_argument('--benchmark', action='store_true', dest="benchmark")
    _root_parser.add_argument('--no-benchmark', action='store_false', dest="benchmark")
    _root_parser.add_argument('--num_io_workers', default=0, type=int)
    _root_parser.add_argument('--k_fold_validation', default=False, type=bool)
    _root_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    _root_parser.add_argument('--seed', type=int, default=0)
    _root_parser.set_defaults(benchmark=True)
    _root_parser.set_defaults(augment=False)

    _model_parser = TrainRTGENE.add_model_specific_args(_root_parser, root_dir)
    _hyperparams = _model_parser.parse_args()

    torch.manual_seed(_hyperparams.seed)
    torch.cuda.manual_seed(_hyperparams.seed)
    np.random.seed(_hyperparams.seed)
    random.seed(_hyperparams.seed)

    if _hyperparams.benchmark:
        torch.backends.cudnn.benchmark = True

    _train_subjects = []
    _valid_subjects = []
    _test_subjects = []
    if _hyperparams.dataset == "rt_gene":
        if _hyperparams.k_fold_validation:
            _train_subjects.append([1, 2, 8, 10, 3, 4, 7, 9])
            _train_subjects.append([1, 2, 8, 10, 5, 6, 11, 12, 13])
            _train_subjects.append([3, 4, 7, 9, 5, 6, 11, 12, 13])
            # validation set is always subjects 14, 15 and 16
            _valid_subjects.append([0, 14, 15, 16])
            _valid_subjects.append([0, 14, 15, 16])
            _valid_subjects.append([0, 14, 15, 16])
            # test subjects
            _test_subjects.append([5, 6, 11, 12, 13])
            _test_subjects.append([3, 4, 7, 9])
            _test_subjects.append([1, 2, 8, 10])
        else:
            _train_subjects.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            _valid_subjects.append([0])  # Note that this is a hack and should not be used to get results for papers
            _test_subjects.append([0])
    else:
        file = h5pickle.File(_hyperparams.hdf5_file, mode="r")
        keys = [int(subject[1:]) for subject in list(file.keys())]
        file.close()
        if _hyperparams.k_fold_validation:
            all_subjects = range(len(keys))
            for leave_one_out_idx in all_subjects:
                _train_subjects.append(all_subjects[:leave_one_out_idx] + all_subjects[leave_one_out_idx + 1:])
                _valid_subjects.append([leave_one_out_idx])  # Note that this is a hack and should not be used to get results for papers
                _test_subjects.append([leave_one_out_idx])
        else:
            _train_subjects.append(keys[1:])
            _valid_subjects.append([keys[0]])
            _test_subjects.append([keys[0]])

    for fold, (train_s, valid_s, test_s) in enumerate(zip(_train_subjects, _valid_subjects, _test_subjects)):
        #skip fold 0
        #if fold ==0:
        #   continue
        complete_path = os.path.abspath(os.path.join(_hyperparams.save_dir, "fold_{}/".format(fold)))

        _model = TrainRTGENE(hparams=_hyperparams, train_subjects=train_s, validate_subjects=valid_s, test_subjects=test_s)
        # save all models
        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(complete_path, "{epoch}-{val_loss:.3f}"), monitor='val_loss', mode='min', verbose=True,
                                              save_top_k=-1 if not _hyperparams.augment else 5)
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, verbose=True, patience=20 if _hyperparams.augment else 13, mode='min')
        # start training
        trainer = Trainer(gpus=_hyperparams.gpu,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop_callback,
                          progress_bar_refresh_rate=1,
                          min_epochs=64 if _hyperparams.augment else 15,
                          max_epochs=100 if _hyperparams.augment else 20,
                          accumulate_grad_batches=_hyperparams.accumulate_grad_batches)
        trainer.fit(_model)
        trainer.test()
