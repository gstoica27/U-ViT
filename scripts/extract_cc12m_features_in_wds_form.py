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


def find_bad_tars(output_dir):
    bad = []
    for fname in tqdm(sorted(os.listdir(output_dir)), desc="Checking tar files"):
        if not fname.endswith(".tar"):
            continue
        path = os.path.join(output_dir, fname)
        try:
            with tarfile.open(path, "r") as tf:
                # iterate through all members to catch truncation errors
                for _ in tf:
                    pass
        except (tarfile.ReadError, EOFError, OSError) as e:
            print(f"[BROKEN] {path} ({type(e).__name__}: {e})")
            bad.append(path)
    return bad


def delete_bad_tars(output_dir, dry_run=False):
    bad = find_bad_tars(output_dir)
    if not bad:
        print("No corrupt .tar files found.")
        return

    print("\nCorrupt tar files:")
    for path in bad:
        print("  ", path)

    if dry_run:
        print("\n[DRY RUN] Not deleting anything. "
              "Call delete_bad_tars(output_dir, dry_run=False) to actually delete.")
        return

    for path in bad:
        print(f"[DELETE] {path}")
        os.remove(path)


def get_existing_keys(output_dir):
    tarfiles = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".tar")
    ]
    
    existing_keys = set()
    if len(tarfiles) == 0:
        return existing_keys
    # pdb.set_trace()
    dataset = wds.WebDataset(tarfiles)
    for sample in tqdm(dataset):
        existing_keys.add(sample["__key__"])
    return existing_keys



def process_and_write_batch(batch, autoencoder, clip_model, preprocessor, tokenizer, device, sink, existing_keys=set()):
    # Separate batch components
    raw_images = [ex[0] for ex in batch]
    cropped_images = [ex[1] for ex in batch]
    captions = [ex[2] for ex in batch]
    keys = [ex[3] for ex in batch]
    urls = [ex[4] for ex in batch]
    local_paths = [ex[5] for ex in batch]
    jsons = [ex[6] for ex in batch]
    
    valid_idxs = [i for i, key in enumerate(keys) if key not in existing_keys]
    if len(valid_idxs) == 0:
        return
    # pdb.set_trace()
    raw_images = [raw_images[i] for i in valid_idxs]
    cropped_images = [cropped_images[i] for i in valid_idxs]
    captions = [captions[i] for i in valid_idxs]
    keys = [keys[i] for i in valid_idxs]
    urls = [urls[i] for i in valid_idxs]
    local_paths = [local_paths[i] for i in valid_idxs]
    jsons = [jsons[i] for i in valid_idxs]

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
    for i in range(len(valid_idxs)):
        try:
            buffer = io.BytesIO()
            raw_images[i].save(buffer, format="JPEG")
            raw_image_bytes = buffer.getvalue()
            buffer.close()
        except:
            print(f"Skipping key={keys[i]} due to image error: {e}")
            continue

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
    existing_keys = get_existing_keys(output_dir)
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

    last_shard_idx = int(sorted(os.listdir(output_dir))[-1].replace("shard-", "").replace(".tar", "")) if len(os.listdir(output_dir)) > 0 else 0
    sink = wds.ShardWriter(os.path.join(output_dir, "shard-%06d.tar"), maxcount=1000, start_shard=last_shard_idx+1)

    with torch.no_grad():
        batch = []
        for _, example in tqdm(enumerate(dataset)):
            batch.append(example)
            
            if len(batch) == batch_size:
                process_and_write_batch(batch, autoencoder, clip_model, preprocessor, tokenizer, device, sink, existing_keys)
                batch = []
        
        # Process remaining batch
        if len(batch) > 0:
            process_and_write_batch(batch, autoencoder, clip_model, preprocessor, tokenizer, device, sink, existing_keys)


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

    delete_bad_tars(args.output_dir)
    # pdb.set_trace()

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