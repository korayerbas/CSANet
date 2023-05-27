from model_csa import CSANET
from load_data import LoadData, LoadVisualData

from torch.utils.data import DataLoader
import torch 
import imageio
import numpy as np
from torchvision import transforms

device = torch.device("cpu")
dslr_scale = float(1) / (2 ** (0 - 1))
dataset_dir='/content/gdrive/MyDrive/ColabNotebooks/pynet_fullres/dataset/'

visual_dataset = LoadVisualData(dataset_dir, 1, dslr_scale)
to_image = transforms.Compose([transforms.ToPILImage()])
visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

generator = CSANET(instance_norm=True, instance_norm_level_1=True).to(device)
generator.load_state_dict(torch.load("/content/gdrive/MyDrive/ColabNotebooks/CSANet/model_csa/85.pth"), strict=False)

generator.eval()
with torch.no_grad():

  visual_iter = iter(visual_loader)
  for j in range(len(visual_loader)):

    torch.cuda.empty_cache()
    raw_image = next(visual_iter)
    raw_image = raw_image.to(device, non_blocking=True)

    enhanced = generator(raw_image.detach())
    enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))
    imageio.imwrite("/content/gdrive/MyDrive/ColabNotebooks/CSANet/results_csa/" + str(j) + "xxx.jpg", enhanced)
