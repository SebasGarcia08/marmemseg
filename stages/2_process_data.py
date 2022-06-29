import os
import os.path as osp
import json
import logging

from marsemseg.datasets import PALETTE

import cv2
from PIL import Image
import hydra 
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

@hydra.main(config_path="../", config_name='config')
def main(cfg):
    args = cfg.process_data
    ann_folder = osp.join(args.dataset_dir, "masks")
    img_folder = osp.join(args.dataset_dir, "images")
    os.makedirs(args.output.ann_dir, exist_ok=True)

    palette_array = np.array(PALETTE, dtype=np.uint8)

    logging.info("Processing annotations...")
    for filename in tqdm(os.listdir(ann_folder), desc="Processing annotations"):
        mask = cv2.imread(osp.join(ann_folder, filename)).astype(np.uint8)
        mask = Image.fromarray(mask[..., 0]).convert("P")
        mask.putpalette(palette_array)
        mask.save(osp.join(args.output.ann_dir, filename))

    means = [0.0, 0.0, 0.0]
    stds = [0.0, 0.0, 0.0]
    train_filenames = []
    logger.info("Reading training filenames...")
    with open(osp.join(cfg.define_splits.output.dir, "train.txt"), "r") as f:
        for line in f:
            train_filenames.append(line.strip() + ".jpg")

    logger.info(f"Calculating means and stds on {len(train_filenames)} training images...")
    for filename in tqdm(train_filenames, desc="Calculating means and stds"):
        img = cv2.imread(osp.join(img_folder, filename))

        for ch in range(img.shape[-1]):
            means[ch] += np.mean(img[..., ch])
            stds[ch] += np.std(img[..., ch])
    
    means = [mean / len(train_filenames) for mean in means]
    stds = [std / len(train_filenames) for std in stds]

    img_norm_cfg = {
        "mean": means,
        "std": stds,
        "to_rgb": True,
    }

    logger.info("Writing img_norm_cfg...")
    with open(osp.join(args.output.dir, "img_norm_cfg.json"), "w") as f:
        json.dump(img_norm_cfg, f)
    logger.info("Done.")


if __name__ == "__main__":
    main()