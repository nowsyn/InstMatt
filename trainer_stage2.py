import os
import numpy as np
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
from   torch.nn import SyncBatchNorm
import torch.optim.lr_scheduler as lr_scheduler
from   torch.nn.parallel import DistributedDataParallel

import utils
from   utils import CONFIG
import networks


class Trainer(object):

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 logger,
                 tb_logger):

        cudnn.benchmark = True

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.tb_logger = tb_logger

        self.cfg = CONFIG
        self.model_config = CONFIG.model
        self.train_config = CONFIG.train
        self.log_config = CONFIG.log
        self.loss_dict = {'rec': None,
                          'comp': None,
                          'smooth_l1':None,
                          'grad':None,
                          'gabor':None,
                          'lap': None,}
        self.test_loss_dict = {'rec': None,
                               'smooth_l1':None,
                               'mse':None,
                               'sad':None,
                               'grad':None,
                               'gabor':None}

        self.grad_filter = torch.tensor(utils.get_gradfilter()).cuda()
        self.gabor_filter = torch.tensor(utils.get_gaborfilter(16)).cuda()

        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                          [4., 16., 24., 16., 4.],
                                          [6., 24., 36., 24., 6.],
                                          [4., 16., 24., 16., 4.],
                                          [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)

        self.one_tensor = torch.ones((CONFIG.model.batch_size, 1, CONFIG.data.crop_size, CONFIG.data.crop_size)).cuda()

        self.build_model()
        self.resume_step = None
        self.best_loss = 1e+8

        utils.print_network(self.G, CONFIG.version)
        if self.train_config.resume_checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.train_config.resume_checkpoint))
            self.restore_model(self.train_config.resume_checkpoint)

        if self.model_config.imagenet_pretrain and self.train_config.resume_checkpoint is None:
            self.logger.info('Load Imagenet Pretrained: {}'.format(self.model_config.imagenet_pretrain_path))
            if self.model_config.arch.encoder == "vgg_encoder":
                utils.load_VGG_pretrain(self.G, self.model_config.imagenet_pretrain_path)
            else:
                utils.load_imagenet_pretrain(self.G, self.model_config.imagenet_pretrain_path)

        if self.model_config.pretrain is not None and self.train_config.resume_checkpoint is None:
            self.logger.info('Load Pretrained: {}'.format(self.model_config.pretrain))
            checkpoint = torch.load(self.model_config.pretrain, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
            state_dict = utils.filter_mismatch_keys(self.G.state_dict(), checkpoint['state_dict'])
            self.G.load_state_dict(state_dict, strict=False)

        if self.model_config.evaluated is not None:
            self.logger.info('Load Evaluated: {}'.format(self.model_config.evaluated))
            checkpoint = torch.load(self.model_config.evaluated, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
            state_dict = utils.filter_mismatch_keys(self.G.state_dict(), checkpoint['state_dict'])
            self.G.load_state_dict(state_dict, strict=True)

    def build_model(self):

        self.G = networks.get_generator(
            self.cfg,
            encoder=self.model_config.arch.encoder,
            decoder=self.model_config.arch.decoder,
            refiner=self.model_config.arch.refiner,
        )

        self.G.cuda()

        if CONFIG.dist:
            self.logger.info("Using pytorch synced BN")
            self.G = SyncBatchNorm.convert_sync_batchnorm(self.G)

        self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                            lr = self.train_config.G_lr,
                                            betas = [self.train_config.beta1, self.train_config.beta2])

        if CONFIG.dist:
            # SyncBatchNorm only supports DistributedDataParallel with single GPU per process
            self.G = DistributedDataParallel(self.G, device_ids=[CONFIG.local_rank], output_device=CONFIG.local_rank)
        else:
            self.G = nn.DataParallel(self.G)

        self.build_lr_scheduler()

    def build_lr_scheduler(self):
        """Build cosine learning rate scheduler."""
        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                          T_max=self.train_config.cosine_period,
                                                          eta_min=self.train_config.min_lr)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()


    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
        self.resume_step = checkpoint['iter']
        self.logger.info('Loading the trained models from step {}...'.format(self.resume_step))
        self.G.load_state_dict(checkpoint['state_dict'], strict=True)

        if not self.train_config.reset_lr:
            if 'opt_state_dict' in checkpoint.keys():
                try:
                    self.G_optimizer.load_state_dict(checkpoint['opt_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
            else:
                self.logger.info('No Optimizer State Loaded!!')

            if 'lr_state_dict' in checkpoint.keys():
                try:
                    self.G_scheduler.load_state_dict(checkpoint['lr_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
        else:
            self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                              T_max=self.train_config.total_step - self.resume_step - 1)

        if 'loss' in checkpoint.keys():
            self.best_loss = checkpoint['loss']

    def test(self):
        self.G.eval()
        test_loss = 0
        log_info = ""

        self.test_loss_dict['mse'] = 0
        self.test_loss_dict['sad'] = 0
        for loss_key in self.loss_dict.keys():
            if loss_key in self.test_loss_dict and self.loss_dict[loss_key] is not None:
                self.test_loss_dict[loss_key] = 0

        with torch.no_grad():
            for image_dict in self.test_dataloader:
                image, alpha, mask = image_dict['image'], image_dict['alpha'], image_dict['mask']
                alpha_shape = image_dict['alpha_shape'][0]
                if 'rescaled_alpha_shape' in image_dict:
                    rescaled_alpha_shape = image_dict['rescaled_alpha_shape'][0]
                else:
                    rescaled_alpha_shape = None
                image = image.cuda()
                alpha = alpha.cuda()
                mask = mask.cuda()

                pred = utils.reduce_dict([self.G(image[i:i+1], mask[i:i+1]) for i in range(mask.size(0))])

                alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
                alpha_pred = alpha_pred_os8.clone().detach()
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
                weight_os4 = weight_os4.max(dim=1, keepdim=True)[0]
                alpha_pred = alpha_pred * (weight_os4<=0).float() + alpha_pred_os4 * (weight_os4>0).float()
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
                weight_os1 = weight_os1.max(dim=1, keepdim=True)[0]
                alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_os1 * (weight_os1>0).float()

                if self.model_config.arch.refiner:
                    alpha_pred_list = self.G.module.forward_refiner(image,
                        alpha_pred.clone().detach(), pred['feature'].clone().detach(), is_training=False)
                    alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_list[-1] * (weight_os1>0).float()

                if rescaled_alpha_shape is None:
                    h, w = alpha_shape
                    alpha_pred = alpha_pred[:, 0:1, :h, :w]
                else:
                    rh, rw = rescaled_alpha_shape
                    alpha_pred = alpha_pred[..., 0:1, :rh, :rw]
                    h, w = alpha_shape
                    alpha_pred = F.interpolate(alpha_pred, (h,w))

                # value of MSE/SAD here is different from test.py and matlab version
                self.test_loss_dict['mse'] += self.mse(alpha_pred, alpha, None)
                self.test_loss_dict['sad'] += self.sad(alpha_pred, alpha, None)

        # reduce losses from GPUs
        if CONFIG.dist:
            self.test_loss_dict = utils.reduce_tensor_dict(self.test_loss_dict, mode='mean')

        """===== Write Log and Tensorboard ====="""
        # stdout log
        for loss_key in self.test_loss_dict.keys():
            if self.test_loss_dict[loss_key] is not None:
                self.test_loss_dict[loss_key] /= len(self.test_dataloader)
                log_info += loss_key.upper()+": {:.4f} ".format(self.test_loss_dict[loss_key])

        self.logger.info("TEST: LOSS: {:.4f} ".format(test_loss)+log_info)
        torch.cuda.empty_cache()


    def train(self):
        data_iter = iter(self.train_dataloader)

        if self.train_config.resume_checkpoint:
            start = self.resume_step + 1
        else:
            start = 0

        moving_max_grad = 0
        moving_grad_moment = 0.999
        max_grad = 0

        for step in range(start, self.train_config.total_step + 1):
            try:
                image_dict = next(data_iter)
            except:
                data_iter = iter(self.train_dataloader)
                image_dict = next(data_iter)

            image, fg, alpha, trimap, mask = image_dict['image'], image_dict['fg'], image_dict['alpha'], image_dict['trimap'], image_dict['mask']

            h, w = image.size(2), image.size(3)
            image = image.cuda()
            alpha = alpha.cuda()
            trimap = trimap.cuda()
            mask = mask.cuda()
            fg = fg.cuda()

            # train() of DistributedDataParallel has no return
            self.G.train()

            if self.model_config.freeze:
                self.G.module.encoder.apply(utils.set_bn_eval)
                self.G.module.aspp.apply(utils.set_bn_eval)
                self.G.module.decoder.apply(utils.set_bn_eval)

            log_info = ""
            loss = 0

            """===== Update Learning Rate ====="""
            if step < self.train_config.warmup_step and self.train_config.resume_checkpoint is None:
                cur_G_lr = utils.warmup_lr(self.train_config.G_lr, step + 1, self.train_config.warmup_step)
                utils.update_lr(cur_G_lr, self.G_optimizer)

            else:
                self.G_scheduler.step()
                cur_G_lr = self.G_scheduler.get_lr()[0]

            """===== Forward G ====="""
            pred = self.G(image, mask)
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

            weight_os8 = utils.get_unknown_tensor(trimap)
            weight_os8[...] = 1

            flag = False
            if step < self.train_config.warmup_step:
                flag = True
                weight_os4 = utils.get_unknown_tensor(trimap)
                weight_os1 = utils.get_unknown_tensor(trimap)
            elif step < self.train_config.warmup_step * 3:
                if random.randint(0,1) == 0:
                    flag = True
                    weight_os4 = utils.get_unknown_tensor(trimap)
                    weight_os1 = utils.get_unknown_tensor(trimap)
                else:
                    weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                    alpha_pred_os4[weight_os4==0] = alpha_pred_os8[weight_os4==0]
                    weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
                    alpha_pred_os1[weight_os1==0] = alpha_pred_os4[weight_os1==0]
            else:
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                alpha_pred_os4[weight_os4==0] = alpha_pred_os8[weight_os4==0]
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
                alpha_pred_os1[weight_os1==0] = alpha_pred_os4[weight_os1==0]

            weight_os1 = weight_os1.max(dim=1, keepdim=True)[0]
            weight_os4 = weight_os4.max(dim=1, keepdim=True)[0]
            weight_os8 = weight_os8.max(dim=1, keepdim=True)[0]

            if self.model_config.arch.refiner is not None:
                alpha_pred_os1_list = self.G.module.forward_refiner(image,
                    alpha_pred_os1.clone().detach(),
                    pred['feature'].clone().detach(), is_training=True)

            """===== Calculate Loss ====="""
            self.clear_loss_dict()
            if not self.model_config.freeze:
                self.compute_loss(alpha_pred_os1, None, None, weight_os1, None, None, image, fg, None, alpha, weight=1.0)
                self.compute_loss(alpha_pred_os4, None, None, weight_os4, None, None, image, fg, None, alpha, weight=0.5)
                self.compute_loss(alpha_pred_os8, None, None, weight_os8, None, None, image, fg, None, alpha, weight=0.5)

            if len(alpha_pred_os1_list)>3:
                alpha_pred_os1_list = random.sample(alpha_pred_os1_list, 3)

            for alpha_pred_os1 in alpha_pred_os1_list:
                self.compute_loss(alpha_pred_os1, None, None, weight_os1, None, None, image, fg, None, alpha, weight=1.0/len(alpha_pred_os1_list))

            for loss_key in self.loss_dict.keys():
                if self.loss_dict[loss_key] is not None and loss_key in ['rec', 'comp', 'lap', 'alpha', 'sparse', 'cross_rec', 'cross_lap']:
                    loss += self.loss_dict[loss_key]

            """===== Back Propagate ====="""
            self.reset_grad()

            loss.backward()

            """===== Clip Large Gradient ====="""
            if self.train_config.clip_grad:
                if moving_max_grad == 0:
                    moving_max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 1e+6)
                    max_grad = moving_max_grad
                else:
                    max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 2 * moving_max_grad)
                    moving_max_grad = moving_max_grad * moving_grad_moment + max_grad * (
                                1 - moving_grad_moment)

            """===== Update Parameters ====="""
            self.G_optimizer.step()

            """===== Write Log and Tensorboard ====="""
            # stdout log
            if step % self.log_config.logging_step == 0:
                # reduce losses from GPUs
                if CONFIG.dist:
                    self.loss_dict = utils.reduce_tensor_dict(self.loss_dict, mode='mean')
                    loss = utils.reduce_tensor(loss)
                # create logging information
                for loss_key in self.loss_dict.keys():
                    if self.loss_dict[loss_key] is not None:
                        log_info += loss_key.upper() + ": {:.4f}, ".format(self.loss_dict[loss_key])

                self.logger.debug("Image tensor shape: {}. Trimap tensor shape: {}".format(image.shape, trimap.shape))
                log_info = "[{}/{}], ".format(step, self.train_config.total_step) + log_info
                log_info += "lr: {:6f}".format(cur_G_lr)
                self.logger.info(log_info)

                # tensorboard
                if step % self.log_config.tensorboard_step == 0 or step == start:  # and step > start:
                    self.tb_logger.scalar_summary('Loss', loss, step)

                    # detailed losses
                    for loss_key in self.loss_dict.keys():
                        if self.loss_dict[loss_key] is not None:
                            self.tb_logger.scalar_summary('Loss_' + loss_key.upper(),
                                                          self.loss_dict[loss_key], step)

                    self.tb_logger.scalar_summary('LearnRate', cur_G_lr, step)

                    if self.train_config.clip_grad:
                        self.tb_logger.scalar_summary('Moving_Max_Grad', moving_max_grad, step)
                        self.tb_logger.scalar_summary('Max_Grad', max_grad, step)

                    image_set = {'image': (utils.normalize_image(image[-1]).data.cpu().numpy()*255).astype(np.uint8),
                                 'mask': (mask[-1].permute(1,0,2).contiguous().view(1,h,-1).data.cpu().numpy()*255).astype(np.uint8),
                                 'alpha': (alpha[-1].permute(1,0,2).contiguous().view(1,h,-1).data.cpu().numpy()*255).astype(np.uint8),
                                 'weight_os1': (weight_os1[-1].permute(1,0,2).contiguous().view(1,h,-1).data.cpu().numpy()*255).astype(np.uint8),
                                 'alpha_pred_os8': (alpha_pred_os8[-1].permute(1,0,2).contiguous().view(1,h,-1).data.cpu().numpy()*255).astype(np.uint8),
                                 'alpha_pred_os4': (alpha_pred_os4[-1].permute(1,0,2).contiguous().view(1,h,-1).data.cpu().numpy()*255).astype(np.uint8),
                                 'alpha_pred_os1': (alpha_pred_os1[-1].permute(1,0,2).contiguous().view(1,h,-1).data.cpu().numpy()*255).astype(np.uint8)}

                    self.tb_logger.image_summary(image_set, step, phase='train')
                    if CONFIG.local_rank == 0:
                        self.tb_logger.writer.flush()


            """===== TEST ====="""
            if self.train_config.val_step > 0:
                if ((step % self.train_config.val_step) == 0 or step == self.train_config.total_step):# and step > start:
                    self.G.eval()
                    test_loss = 0
                    log_info = ""

                    self.test_loss_dict['mse'] = 0
                    self.test_loss_dict['sad'] = 0
                    for loss_key in self.loss_dict.keys():
                        if loss_key in self.test_loss_dict and self.loss_dict[loss_key] is not None:
                            self.test_loss_dict[loss_key] = 0

                    print('##################### test begin #########################')

                    with torch.no_grad():
                        for tid, image_dict in enumerate(self.test_dataloader):

                            image, alpha, mask = image_dict['image'], image_dict['alpha'], image_dict['mask']
                            alpha_shape = image_dict['alpha_shape'][0]
                            if 'rescaled_alpha_shape' in image_dict:
                                rescaled_alpha_shape = image_dict['rescaled_alpha_shape'][0]
                            else:
                                rescaled_alpha_shape = None
                            image = image.cuda()
                            alpha = alpha.cuda()
                            mask = mask.cuda()

                            pred = utils.reduce_dict([self.G(image[i:i+1], mask[i:i+1]) for i in range(mask.size(0))])

                            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
                            alpha_pred = alpha_pred_os8.clone().detach()
                            weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
                            weight_os4 = weight_os4.max(dim=1, keepdim=True)[0]
                            alpha_pred = alpha_pred * (weight_os4<=0).float() + alpha_pred_os4 * (weight_os4>0).float()
                            weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
                            weight_os1 = weight_os1.max(dim=1, keepdim=True)[0]
                            alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_os1 * (weight_os1>0).float()

                            if self.model_config.arch.refiner is not None:
                                alpha_pred_list = self.G.module.forward_refiner(image,
                                                  alpha_pred.clone().detach(), pred['feature'].clone().detach(),
                                                  is_training=False)
                                alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_list[-1] * (weight_os1>0).float()

                            if rescaled_alpha_shape is None:
                                h, w = alpha_shape
                                alpha_pred = alpha_pred[:, 0:1, :h, :w]
                            else:
                                rh, rw = rescaled_alpha_shape
                                alpha_pred = alpha_pred[..., 0:1, :rh, :rw]
                                h, w = alpha_shape
                                alpha_pred = F.interpolate(alpha_pred, (h,w))

                            # value of MSE/SAD here is different from test.py and matlab version
                            self.test_loss_dict['mse'] += self.mse(alpha_pred, alpha, None)
                            self.test_loss_dict['sad'] += self.sad(alpha_pred, alpha, None)

                            if self.train_config.rec_weight > 0:
                                self.test_loss_dict['rec'] += self.regression_loss(alpha_pred, alpha) \
                                                            * self.train_config.rec_weight

                    print('##################### test end #########################')

                    # reduce losses from GPUs
                    if CONFIG.dist:
                        self.test_loss_dict = utils.reduce_tensor_dict(self.test_loss_dict, mode='mean')

                    """===== Write Log and Tensorboard ====="""
                    # stdout log
                    for loss_key in self.test_loss_dict.keys():
                        if self.test_loss_dict[loss_key] is not None:
                            self.test_loss_dict[loss_key] /= len(self.test_dataloader)
                            # logging
                            log_info += loss_key.upper()+": {:.4f} ".format(self.test_loss_dict[loss_key])
                            self.tb_logger.scalar_summary('Loss_'+loss_key.upper(),
                                                        self.test_loss_dict[loss_key], step, phase='test')

                            if loss_key in ['rec']:
                                test_loss += self.test_loss_dict[loss_key]

                    self.logger.info("TEST: LOSS: {:.4f} ".format(test_loss)+log_info)
                    self.tb_logger.scalar_summary('Loss', test_loss, step, phase='test')

                    image_set = {'image': (utils.normalize_image(image[0, ...]).data.cpu().numpy() * 255).astype(np.uint8),
                                 'mask': (mask[0, 0:1, ...].data.cpu().numpy() * 255).astype(np.uint8),
                                 'alpha': (alpha[0, 0:1, ...].data.cpu().numpy() * 255).astype(np.uint8),
                                 'alpha_pred': (alpha_pred[0, 0:1, ...].data.cpu().numpy() * 255).astype(np.uint8)}

                    self.tb_logger.image_summary(image_set, step, phase='test')
                    if CONFIG.local_rank == 0:
                        self.tb_logger.writer.flush()

                """===== Save Model ====="""
                if (step % self.log_config.checkpoint_step == 0 or step == self.train_config.total_step) \
                        and CONFIG.local_rank == 0 and (step > start):
                    self.logger.info('Saving the trained models from step {}...'.format(iter))
                    self.save_model("latest_model", step, loss)
                    if self.test_loss_dict['mse'] is not None and self.test_loss_dict['mse'] < self.best_loss:
                        self.best_loss = self.test_loss_dict['mse']
                        self.save_model("best_model", step, loss)
            else:
                """===== Save Model ====="""
                if (step % self.log_config.checkpoint_step == 0 or step == self.train_config.total_step) \
                        and CONFIG.local_rank == 0 and (step > start):
                    self.logger.info('Saving the trained models from step {}...'.format(iter))
                    self.save_model("latest_model", step, loss)

            torch.cuda.empty_cache()


    def save_model(self, checkpoint_name, iter, loss):
        """Restore the trained generator and discriminator."""
        torch.save({
            'iter': iter,
            'loss': loss,
            'state_dict': self.G.state_dict(),
            'opt_state_dict': self.G_optimizer.state_dict(),
            'lr_state_dict': self.G_scheduler.state_dict()
        }, os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(checkpoint_name)))

    def clear_loss_dict(self):
        for key in ['rec', 'lap', 'comp', 'alpha']:
            self.loss_dict[key] = 0.0

    def compute_loss(self, alpha_pred_os1, alpha_pred_os4, alpha_pred_os8, weight_os1, weight_os4, weight_os8, image, fg, bg, alpha, weight):
        if self.train_config.rec_weight > 0:
            self.loss_dict['rec'] += (
                self.regression_loss(alpha_pred_os1, alpha, loss_type='l1', weight=weight_os1)) * \
                self.train_config.rec_weight * weight

        if self.train_config.lap_weight > 0:
            self.loss_dict['lap'] += (
                self.lap_loss(logit=alpha_pred_os1, target=alpha, gauss_filter=self.gauss_filter.repeat(3,1,1,1),
                    loss_type='l1', weight=weight_os1)) * \
                self.train_config.lap_weight * weight

        if self.train_config.comp_weight > 0:
            self.loss_dict['comp'] += (
                self.ch3_composition_loss(alpha_pred_os1, fg, image, weight=weight_os1)) * \
                self.train_config.comp_weight * weight

        if self.train_config.alpha_weight > 0:
            self.loss_dict['alpha'] += (
                self.alpha_constraint_loss(alpha_pred_os1, self.one_tensor, weight=weight_os1)) * \
                self.train_config.alpha_weight * weight


    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None):
        """
        Alpha reconstruction loss
        :param logit:
        :param target:
        :param loss_type: "l1" or "l2"
        :param weight: tensor with shape [N,1,H,W] weights for each pixel
        :return:
        """
        if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
        else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


    @staticmethod
    def smooth_l1(logit, target, weight):
        loss = torch.sqrt((logit * weight - target * weight)**2 + 1e-6)
        loss = torch.sum(loss) / (torch.sum(weight) + 1e-8)
        return loss


    @staticmethod
    def mse(logit, target, weight):
        return Trainer.regression_loss(logit, target, loss_type='l2', weight=weight)

    @staticmethod
    def sad(logit, target, weight):
        if weight is None:
            return F.l1_loss(logit, target, reduction='sum') / 1000
        else:
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / 1000

    @staticmethod
    def ch3_composition_loss(alpha, fg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = torch.sum(fg * alpha.unsqueeze(2), dim=1)
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight)

    @staticmethod
    def naive_composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = fg * alpha + bg * (1 - alpha)
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight)

    @staticmethod
    def extend_composition_loss(batch_size, alpha, alpha_gt, fg, bg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = utils.group_reduce_sum(fg * alpha_gt, batch_size) - fg * alpha_gt + fg * alpha + \
                 bg * (1-alpha-utils.group_reduce_sum(alpha_gt, batch_size) + alpha_gt)
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight) / 3.

    @staticmethod
    def composition_loss(batch_size, alpha, alpha_bg, fg, bg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = utils.group_reduce_sum(fg * alpha, batch_size) + bg * alpha_bg
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight) / 3. / (alpha.size(0) / float(batch_size))

    @staticmethod
    def alpha_constraint_loss(alpha, one_tensor, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        return Trainer.regression_loss(alpha_sum, one_tensor, loss_type=loss_type, weight=weight)

    @staticmethod
    def sparse_loss(alpha, loss_type='l0', weight=None):
        """
        Alpha composition loss
        """
        if weight is None:
            l0norm = torch.norm(alpha, p=0, dim=0, keepdim=True)
        else:
            l0norm = torch.norm(alpha * weight, p=0, dim=0, keepdim=True)
        mask = ((l0norm - 2)>0).float().detach()
        return ((l0norm - 2) * mask).sum() / mask.sum()

    @staticmethod
    def gabor_loss(logit, target, gabor_filter, loss_type='l2', weight=None):
        """ pass """
        gabor_logit = F.conv2d(logit, weight=gabor_filter, padding=2)
        gabor_target = F.conv2d(target, weight=gabor_filter, padding=2)

        return Trainer.regression_loss(gabor_logit, gabor_target, loss_type=loss_type, weight=weight)

    @staticmethod
    def grad_loss(logit, target, grad_filter, loss_type='l1', weight=None):
        """ pass """
        grad_logit = F.conv2d(logit, weight=grad_filter, padding=1)
        grad_target = F.conv2d(target, weight=grad_filter, padding=1)
        grad_logit = torch.sqrt((grad_logit * grad_logit).sum(dim=1, keepdim=True) + 1e-8)
        grad_target = torch.sqrt((grad_target * grad_target).sum(dim=1, keepdim=True) + 1e-8)

        return Trainer.regression_loss(grad_logit, grad_target, loss_type=loss_type, weight=weight)

    @staticmethod
    def lap_loss(logit, target, gauss_filter, loss_type='l1', weight=None):
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''
        def conv_gauss(x, kernel):
            x = F.pad(x, (2,2,2,2), mode='reflect')
            x = F.conv2d(x, kernel, groups=x.shape[1])
            return x

        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x, kernel):
            N, C, H, W = x.shape
            cc = torch.cat([x, torch.zeros(N,C,H,W).cuda()], dim = 3)
            cc = cc.view(N, C, H*2, W)
            cc = cc.permute(0,1,3,2)
            cc = torch.cat([cc, torch.zeros(N, C, W, H*2).cuda()], dim = 3)
            cc = cc.view(N, C, W*2, H*2)
            x_up = cc.permute(0,1,3,2)
            return conv_gauss(x_up, kernel=4*gauss_filter)

        def lap_pyramid(x, kernel, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                filtered = conv_gauss(current, kernel)
                down = downsample(filtered)
                up = upsample(down, kernel)
                diff = current - up
                pyr.append(diff)
                current = down
            return pyr

        def weight_pyramid(x, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                down = downsample(current)
                pyr.append(current)
                current = down
            return pyr

        pyr_logit = lap_pyramid(x = logit, kernel = gauss_filter, max_levels = 5)
        pyr_target = lap_pyramid(x = target, kernel = gauss_filter, max_levels = 5)
        if weight is not None:
            pyr_weight = weight_pyramid(x = weight, max_levels = 5)
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
        else:
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target)))
