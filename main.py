import os
import toml
import argparse
from pprint import pprint

import torch
from torch.utils.data import DataLoader

import utils
from utils import CONFIG
from dataloader.offline_image_file import CustomImageFileTrain, CustomImageFileTest
from dataloader.prefetcher import Prefetcher


def main(args):

    if args.stage == 1:
        from trainer_stage1 import Trainer
        from dataloader.offline_data_generator_stage1 import CustomDataGenerator, batch_collator
    else:
        from trainer_stage2 import Trainer
        from dataloader.offline_data_generator_stage2 import CustomDataGenerator, batch_collator

    # Train or Test
    if CONFIG.phase.lower() == "train":
        # set distributed training
        if CONFIG.dist:
            CONFIG.gpu = CONFIG.local_rank
            torch.cuda.set_device(CONFIG.gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            CONFIG.world_size = torch.distributed.get_world_size()

        # Create directories if not exist.
        if CONFIG.local_rank == 0:
            utils.make_dir(CONFIG.log.logging_path)
            utils.make_dir(CONFIG.log.tensorboard_path)
            utils.make_dir(CONFIG.log.checkpoint_path)
        # Create a logger
        logger, tb_logger = utils.get_logger(CONFIG.log.logging_path,
                                             CONFIG.log.tensorboard_path,
                                             logging_level=CONFIG.log.logging_level)
        train_image_file = CustomImageFileTrain('train',
                                          alpha_dir=CONFIG.data.train_alpha,
                                          merged_dir=CONFIG.data.train_merged,
                                          mask_dir=CONFIG.data.train_mask,
                                          fg_dir=CONFIG.data.train_fg,
                                          bg_dir=CONFIG.data.train_bg)
        test_image_file = CustomImageFileTest('val',
                                        alpha_dir=CONFIG.data.test_alpha,
                                        merged_dir=CONFIG.data.test_merged,
                                        mask_dir=CONFIG.data.test_mask)

        train_dataset = CustomDataGenerator(train_image_file, phase='train')
        test_dataset = CustomDataGenerator(test_image_file, phase='val')

        if CONFIG.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            train_sampler = None
            test_sampler = None

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CONFIG.model.batch_size,
                                      shuffle=(train_sampler is None),
                                      num_workers=CONFIG.data.workers,
                                      pin_memory=True,
                                      collate_fn=batch_collator,
                                      sampler=train_sampler,
                                      drop_last=True)
        train_dataloader = Prefetcher(train_dataloader)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=CONFIG.data.workers,
                                     collate_fn=batch_collator,
                                     sampler=test_sampler,
                                     drop_last=False)

        trainer = Trainer(train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          logger=logger,
                          tb_logger=tb_logger)

        if args.evaluate:
            trainer.test()
        else:
            trainer.train()
    else:
        raise NotImplementedError("Unknown Phase: {}".format(CONFIG.phase))


if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--stage', type=int, default=1, choices=[1,2])
    parser.add_argument('--config', type=str, default='config/gca-dist.toml')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--evaluate', action="store_true")

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    CONFIG.log.tensorboard_path = os.path.join(CONFIG.log.tensorboard_path, CONFIG.version)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.checkpoint_path, CONFIG.version)
    if args.local_rank == 0:
        print('CONFIG: ')
        pprint(CONFIG)
    CONFIG.local_rank = args.local_rank

    # Train
    main(args)
