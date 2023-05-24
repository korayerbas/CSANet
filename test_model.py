# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import numpy as np
import sys
import os

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from load_data import LoadVisualData
from model_csa import CSANET
import utils
import imageio

to_image = transforms.Compose([transforms.ToPILImage()])

restore_epoch, dataset_dir, use_gpu, orig_model = utils.process_test_model_args(sys.argv)
dslr_scale = 1


def test_model():

    if use_gpu == "true":
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = torch.device("cpu")

    # Creating dataset loaders

    visual_dataset = LoadVisualData(dataset_dir, 2, dslr_scale, full_resolution=True)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # Creating and loading pre-trained PyNET model

    model = CSANET(instance_norm=True, instance_norm_level_1=True).to(device)
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load("C:\\PYNET\\models\\csanet" + "_epoch_" + str(restore_epoch) + ".pth"), strict=True)

    if use_gpu == "true":
        model.half()

    model.eval()

    # Processing full-resolution RAW images

    with torch.no_grad():

        visual_iter = iter(visual_loader)
        for j in range(len(visual_loader)):

            print("Processing image " + str(j))

            torch.cuda.empty_cache()
            raw_image = next(visual_iter)

            if use_gpu == "true":
                raw_image = raw_image.to(device, dtype=torch.half)
            else:
                raw_image = raw_image.to(device)

            # Run inference

            enhanced = model(raw_image.detach())
            enhanced = np.asarray(to_image(torch.squeeze(enhanced.float().detach().cpu())))

            # Save the results as .png images
            imageio.imwrite("C:\\PYNET\\dataset\\full_res_results" + str(j) + "_epoch_" + str(restore_epoch) + ".png", enhanced)


if __name__ == '__main__':
    test_model()
