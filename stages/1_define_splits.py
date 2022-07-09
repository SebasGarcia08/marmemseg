import os
import os.path as osp
import random

import numpy as np
import pandas as pd
import hydra
import mmcv


@hydra.main(config_path="../", config_name='params')
def main(cfg):
    args = cfg.define_splits
    random.seed(cfg.base.random.seed)

    images_dir = osp.join(args.dataset_dir, "images")
    mmcv.mkdir_or_exist(args.output.dir)
    filename_list = [
        osp.splitext(filename)[0]
        for filename in mmcv.scandir(images_dir, suffix=".jpg")
    ]
    random.shuffle(filename_list)
    train_size = int(len(filename_list) * args.train_ratio)
    train_filenames = filename_list[:train_size]
    val_filenames = filename_list[train_size:]

    with open(osp.join(args.output.dir, "train.txt"), "w") as f:
        for filename in train_filenames:
            f.write(filename + "\n")

    with open(osp.join(args.output.dir, "val.txt"), "w") as f:
        for filename in val_filenames:
            f.write(filename + "\n")
    

if __name__ == "__main__":
    main()