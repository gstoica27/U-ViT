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


def process_and_write_batch(batch, autoencoder, clip_model, preprocessor, tokenizer, device, sink):
    # Separate batch components
    raw_images = [ex[0] for ex in batch]
    cropped_images = [ex[1] for ex in batch]
    captions = [ex[2] for ex in batch]
    keys = [ex[3] for ex in batch]
    urls = [ex[4] for ex in batch]
    local_paths = [ex[5] for ex in batch]
    jsons = [ex[6] for ex in batch]
    # pdb.set_trace()
    # Batch process cropped images through autoencoder
    cropped_images_batch = []
    for cropped_image in cropped_images:
        if len(cropped_image.shape) == 3:
            cropped_image = cropped_image[None, ...]
        cropped_images_batch.append(torch.from_numpy(cropped_image))
    cropped_images_batch = torch.cat(cropped_images_batch, dim=0).to(device)
    # pdb.set_trace()
    # Encode moments in batch
    moments_batch = autoencoder(cropped_images_batch, fn="encode_moments").detach().cpu().numpy()
    cropped_images_batch_cpu = cropped_images_batch.detach().cpu().numpy()
    
    # Batch process images through CLIP
    preprocessed_images = torch.stack([preprocessor(img) for img in raw_images]).to(device)
    image_encodings_batch = clip_model.encode_image(preprocessed_images, normalize=True).detach().cpu().numpy()
    
    # Batch process text through CLIP
    tokenized_captions = tokenizer(captions).to(device)
    text_encodings_batch = clip_model.encode_text(tokenized_captions, normalize=True).detach().cpu().numpy()
    text_hiddens_batch = clip_model.forward_intermediates(
        text=tokenized_captions, intermediates_only=True, normalize_intermediates=False
    )['text_intermediates'][-1].detach().cpu().numpy()
    
    # Write individual samples
    for i in range(len(batch)):
        buffer = io.BytesIO()
        raw_images[i].save(buffer, format="JPEG")
        raw_image_bytes = buffer.getvalue()
        buffer.close()

        save_dict = {
            "raw_image.jpg": raw_image_bytes,
            "cropped_image.npy": cropped_images_batch_cpu[i],
            "caption.txt": captions[i],
            "moments.npy": moments_batch[i],
            "text_hidden.npy": text_hiddens_batch[i],
            "image_encoding.npy": image_encodings_batch[i],
            "text_encoding.npy": text_encodings_batch[i],
            "__key__": keys[i],
            "__url__": urls[i],
            "__local_path__": local_paths[i],
            "json.json": jsons[i]
        }
        sink.write(save_dict)


def preprocess_data(tarfiles, output_dir, clip_model_name='hf-hub:timm/ViT-SO400M-16-SigLIP2-384', resolution=256, batch_size=32):
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
        batch = []
        for _, example in tqdm(enumerate(dataset)):
            batch.append(example)
            
            if len(batch) == batch_size:
                process_and_write_batch(batch, autoencoder, clip_model, preprocessor, tokenizer, device, sink)
                batch = []
        
        # Process remaining batch
        if len(batch) > 0:
            process_and_write_batch(batch, autoencoder, clip_model, preprocessor, tokenizer, device, sink)


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
    parser.add_argument(
        "--batch-size", type=int, default=32, 
        help="Batch size for processing"
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
        resolution=args.resolution,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()