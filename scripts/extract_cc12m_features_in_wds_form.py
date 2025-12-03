import os
import io
import pdb
import sys
import torch
import tarfile
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import libs.autoencoder
import webdataset as wds
from datasets import CCDataset
from open_clip import create_model_from_pretrained, get_tokenizer


def detach_and_to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def preprocess_data(tarfiles, output_dir, clip_model_name='hf-hub:timm/ViT-SO400M-16-SigLIP2-384', resolution=256):
    dataset = CCDataset(path=tarfiles, resolution=resolution)
    device = "cuda"
    os.makedirs(output_dir, exist_ok=True)
    autoencoder = libs.autoencoder.get_model(
        '/weka/prior-default/georges/checkpoints/uvit_repo/assets/stable-diffusion/autoencoder_kl.pth'
    )
    autoencoder = autoencoder.to(device)

    clip_model, preprocessor = create_model_from_pretrained(clip_model_name)
    clip_model = clip_model.eval().to(device)
    tokenizer = get_tokenizer(clip_model_name)

    sink = wds.ShardWriter(os.path.join(output_dir, "shard-%06d.tar"), maxcount=1000)

    with torch.no_grad():
        for _, example in tqdm(enumerate(dataset)):
            raw_image, cropped_image, caption, key, url, local_path, json_ = example
            # pdb.set_trace()
            if len(cropped_image.shape) == 3:
                cropped_image = cropped_image[None, ...]
            cropped_image = torch.from_numpy(cropped_image).to(device)
            moments = autoencoder(cropped_image, fn="encode_moments").squeeze(0).detach().cpu().numpy()
            cropped_image = detach_and_to_cpu(cropped_image)
            
            image_encoding = detach_and_to_cpu(
                clip_model.encode_image(preprocessor(raw_image)[None].to(device), normalize=True)
            )
            tokenized_caption = tokenizer([caption]).to(device)
            text_encoding = detach_and_to_cpu(clip_model.encode_text(tokenized_caption, normalize=True))
            text_hidden = detach_and_to_cpu(clip_model.forward_intermediates(
                text=tokenized_caption, intermediates_only=True, normalize_intermediates=False
            )['text_intermediates'][-1])
            
            buffer = io.BytesIO()
            raw_image.save(buffer, format="JPEG")
            raw_image_bytes = buffer.getvalue()
            buffer.close()

            save_dict = {
                "raw_image.jpg": raw_image_bytes,
                "cropped_image.npy": cropped_image,
                "caption.txt": caption,
                "moments.npy": moments,
                "text_hidden.npy": text_hidden,
                "image_encoding.npy": image_encoding,
                "text_encoding.npy": text_encoding,
                "__key__": key,
                "__url__": url,
                "__local_path__": local_path,
                "json": json_
            }
            sink.write(save_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir", type=str, required=True, 
        help="Directory containing CC12M tar files for a specific split (train/val)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, 
        help="Directory to save the extracted features"
    )
    parser.add_argument(
        "--shard-start", type=int, required=True, 
        help="Starting shard index"
    )
    parser.add_argument(
        "--shard-end", type=int, required=True, 
        help="Ending shard index"
    )
    parser.add_argument(
        "--resolution", type=int, default=256, 
        help="Image resolution for cropping"
    )
    parser.add_argument(
        "--clip-model-name", type=str, default='hf-hub:timm/ViT-SO400M-16-SigLIP2-384', 
        help="CLIP model name for feature extraction"
    )
    args = parser.parse_args()

    tarfiles = [
        os.path.join(args.split_dir, f"shard_{i:06d}.tar") for i in range(
            args.shard_start, args.shard_end + 1
        )
    ]
    preprocess_data(
        tarfiles, args.output_dir, 
        clip_model_name=args.clip_model_name, 
        resolution=args.resolution
    )


if __name__ == "__main__":
    main()