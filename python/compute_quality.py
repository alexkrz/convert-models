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

from impl_pytorch.cr_fiqa.backbones.iresnet import iresnet50 as crfiqa_iresnet50
from impl_pytorch.cr_fiqa.backbones.iresnet import iresnet100 as crfiqa_iresnet100
from impl_pytorch.magface.network_inf import builder_inf as magface_builder_inf
from impl_pytorch.sdd_fiqa.generate_pseudo_labels.extract_embedding.model.model import (
    R50 as sddfiqa_r50,
)
from impl_pytorch.ser_fiq.backbones.iresnet import iresnet18 as serfiq_iresnet18
from impl_pytorch.ser_fiq.backbones.iresnet import iresnet50 as serfiq_iresnet50

AVAIL_METHODS = [
    "sdd-fiqa",
    "crfiqa-s",
    "crfiqa-l",
    "magface",
    "ser-fiq",
]


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
            print("Rezising image")
            img = self.resize_transform(img)
        # Apply additional transforms
        if self.transform:
            img = self.transform(img)
        return img, str(relative_path)


def main(
    data_dir: str = os.environ["DATASET_DIR"] + "/Multi-PIE/aligned_probes",
    data_name: str = "multi-pie",
    file_ext: str = ".png",
    method_name: str = "ser-fiq",
    checkpoint_fp: str = "checkpoints/ser_fiq/resnet18.pth",
    save_dir: str = "results",
    gpu: Optional[int] = 0,
):
    assert method_name in AVAIL_METHODS
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
    print(f"Loading model for {method_name}..")
    if method_name == "sdd-fiqa":
        model = sddfiqa_r50([112, 112], use_type="Qua")
        model_dict = model.state_dict()
        data_dict = {
            key.replace("module.", ""): value for key, value in torch.load(checkpoint_fp).items()
        }
        model_dict.update(data_dict)
        model.load_state_dict(model_dict)
    elif "crfiqa" in method_name:
        if method_name == "crfiqa-s":
            model = crfiqa_iresnet50(num_features=512, qs=1, use_se=False)
        if method_name == "crfiqa-l":
            model = crfiqa_iresnet100(num_features=512, qs=1, use_se=False)
        weight = torch.load(checkpoint_fp)
        model.load_state_dict(weight)
    elif method_name == "magface":
        model = magface_builder_inf(arch="iresnet100", emb_size=512, checkpoint_fp=checkpoint_fp)
    elif method_name == "ser-fiq":
        model = serfiq_iresnet18(dropout=0.4, num_features=512, use_se=False)
        weight = torch.load(checkpoint_fp)
        model.load_state_dict(weight)
    else:
        raise NotImplementedError()

    model.to(device)
    model.eval()

    # Run inference on dataset items
    qs_score_list = []
    filename_list = []
    print("Computing quality scores..")
    for batch_data in tqdm(dataloader):
        imgs, filenames = batch_data
        with torch.no_grad():
            if method_name == "sdd-fiqa":
                qs = model(imgs.to(device))
            elif "crfiqa" in method_name:
                feats, qs = model(imgs.to(device))
            elif method_name == "magface":
                feats = model(imgs.to(device))
                # Use magnitude of the features as quality score
                qs = torch.linalg.norm(feats, axis=1)
            elif method_name == "ser-fiq":
                qs = model.calculate_serfiq(imgs.to(device), T=10, scaling=5.0)
            else:
                raise NotImplementedError()
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
        save_dir / f"{method_name}_{data_name}.csv",
        float_format="%.6f",
        header=True,
        index=False,
        sep=";",
    )


if __name__ == "__main__":
    CLI(main, as_positional=False, parser_mode="omegaconf")
