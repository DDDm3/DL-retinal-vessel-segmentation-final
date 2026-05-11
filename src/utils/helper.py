import numpy as np
import os
import tqdm
import torch
EXPORT_BASE = "../Drive_processed_dataset"

def normalize_mask(mask, **kwargs):
    return mask.astype(np.float32) / 255.0

def tensor_to_image(tensor):
    img = tensor.numpy().transpose(1, 2, 0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    
    img = np.clip(img, 0, 1)
    return img

def export_dataset_to_disk(dataset, split_name):
    print(f"Exporting {split_name} ({len(dataset)} samples)...")
    
    split_dir = os.path.join(EXPORT_BASE, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        save_dict = {
            "image": sample[0],
            "mask": sample[1],
            "manual": sample[2]
        }
        
        # tập Test
        if len(sample) == 4:
            save_dict["manual_2"] = sample[3]
            
        # Lưu file dưới dạng .pt (PyTorch Data)
        save_path = os.path.join(split_dir, f"sample_{i}.pt")
        torch.save(save_dict, save_path)