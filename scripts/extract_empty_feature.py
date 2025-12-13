# import torch
import os
import numpy as np
# import libs.autoencoder
# import libs.clip
# from datasets import MSCOCODatabase
# import argparse
# from tqdm import tqdm

from open_clip import create_model_from_pretrained, get_tokenizer

def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    # clip = libs.clip.FrozenCLIPEmbedder()
    # clip.eval()
    # clip.to(device)
    # # save_dir = f'/weka/prior-default/georges/datasets/mscoco256_features'
    # save_dir = '/weka/oe-training-default/georges/datasets/mscoco256_featuresv2'
    # latent = clip.encode(prompts)
    # print(latent.shape)
    # c = latent[0].detach().cpu().numpy()
    # np.save(os.path.join(save_dir, f'empty_context.npy'), c)

    save_dir = '/weka/oe-training-default/georges/datasets/cc12m-wds-splits'
    clip_model_name = 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384'
    clip_model, preprocessor = create_model_from_pretrained(clip_model_name)
    clip_model = clip_model.eval().to(device)
    tokenizer = get_tokenizer(clip_model_name)


    tokenized_captions = tokenizer(prompts).to(device)
    text_hiddens_batch = clip_model.forward_intermediates(
        text=tokenized_captions, intermediates_only=True, normalize_intermediates=False
    )['text_intermediates'][-1].detach().cpu().numpy()
    print(text_hiddens_batch.shape)
    t = text_hiddens_batch[0]
    np.save(os.path.join(save_dir, f'empty_text_hidden.npy'), t)


if __name__ == '__main__':
    main()
