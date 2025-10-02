import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from jsonargparse import CLI
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.utils import load_model  # isort: skip


class FaceDataset(Dataset):
    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(
        self,
        root_dir: Path,
        file_ext: str = ".jpg",
        img_shape: Tuple[int, int] = (112, 112),
        transform: Optional[transforms.Compose] = default_transform,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.file_list = sorted(list(root_dir.rglob("*" + file_ext)))
        self.img_shape = img_shape
        self.resize_transform = transforms.Resize(self.img_shape)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_p = Path(self.file_list[idx])
        relative_path = file_p.relative_to(self.root_dir)
        img = Image.open(file_p)
        # Check image shape and use resize transform if necessary
        if not img.size == self.img_shape:
            # print("Rezising image")
            img = self.resize_transform(img)
        # Apply additional transforms
        if self.transform:
            img = self.transform(img)
        return img, str(relative_path)


def main(
    data_dir: str = "data/lfw-deepfunneled",
    data_name: str = "lfw",
    file_ext: str = ".jpg",
    method_name: str = "debfiqa_26",
    checkpoint_fp: str = "checkpoints/pytorch/DEBFIQA/debfiqa_26.pth",
    save_dir: str = "results",
    gpu: Optional[int] = 0,
):
    data_dir = Path(data_dir)  # type: Path
    assert data_dir.exists()
    save_dir = Path(save_dir)  # type: Path
    if not save_dir.exists():
        save_dir.mkdir()

    # Assign device where code is executed
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # Only show pytorch the selected GPU
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")  # Use CPU

    # Construct dataset
    print(f"Generating dataset class for {data_name}..")
    dataset = FaceDataset(data_dir, file_ext)
    print("Number of items in dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    # Load model
    model = load_model(method_name, checkpoint_fp)
    model.to(device)
    model.eval()

    # Run inference on dataset items
    qs_score_list = []
    filename_list = []
    print("Computing quality scores..")
    for batch_data in tqdm(dataloader):
        imgs, filenames = batch_data
        with torch.no_grad():
            feats, qs = model(imgs.to(device))
        qs_scores = qs.cpu().numpy()  # Move qs_scores to CPU and convert to numpy array
        filename_list += filenames
        qs_score_list.append(qs_scores)

    print("Saving results..")
    qs_scores_arr = np.concatenate(qs_score_list, axis=0)
    qs_scores_arr = qs_scores_arr.squeeze()
    print("Shape of qs_scores_arr:", qs_scores_arr.shape)
    filename_arr = np.array(filename_list)

    df_out = pd.DataFrame({"sample": filename_arr, "quality": qs_scores_arr})
    print(df_out.head())
    df_out.to_csv(
        save_dir / f"{method_name}_{data_name}.txt",
        float_format="%.6f",
        header=True,
        index=False,
        sep=" ",
    )


if __name__ == "__main__":
    CLI(main, as_positional=False, parser_mode="omegaconf")
