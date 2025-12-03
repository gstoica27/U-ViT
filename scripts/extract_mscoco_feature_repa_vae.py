import pdb
import torch
import os
import numpy as np
import sys
# sys.path.append("/weka/prior-default/georges/research/U-ViT/libs")
# import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm
from PIL import Image
from scripts.repa_vae import StabilityVAEEncoder


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--model_url', default='stabilityai/sd-vae-ft-mse')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MSCOCODatabase(
            root='/weka/prior-default/georges/datasets/mscoco/train2014',
            annFile='/weka/prior-default/georges/datasets/mscoco/annotations/captions_train2014.json',
            size=resolution
        )
        save_dir = f'/weka/oe-training-default/georges/datasets/mscoco{resolution}_featuresv3_repa_vae/train'
    elif args.split == "val":
        datas = MSCOCODatabase(root='/weka/prior-default/georges/datasets/mscoco/val2014',
                             annFile='/weka/prior-default/georges/datasets/mscoco/annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'/weka/oe-training-default/georges/datasets/mscoco{resolution}_featuresv3_repa_vae/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir)

    # autoencoder = libs.autoencoder.get_model('/weka/prior-default/georges/checkpoints/uvit_repo/assets/stable-diffusion/autoencoder_kl.pth')
    # autoencoder.to(device)
    vae = StabilityVAEEncoder(vae_name=args.model_url, batch_size=1)
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            img, x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            mean_std = vae.encode_pixels(x)[0].cpu()
            # pdb.set_trace()
            # moments = autoencoder(x, fn='encode_moments').squeeze(0)
            # moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), mean_std)
            # import pdb; pdb.set_trace()
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(save_dir, f'{idx}.png'))

            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)


if __name__ == '__main__':
    main()
