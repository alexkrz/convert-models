import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main(
    data_dir: str = "data/lfw-deepfunneled",
    data_name: str = "lfw",
    file_ext: str = ".jpg",
    method_name: str = "debfiqa_26",
    checkpoint_fp: str = "checkpoints/onnx/debfiqa_26.onnx",
    save_dir: str = "results",
):
    # Check directories
    data_dir = Path(data_dir)  # type: Path
    save_dir = Path(save_dir)  # type: Path
    if not save_dir.exists():
        save_dir.mkdir()

    # Glob data_dir
    img_list = sorted(list(data_dir.rglob(f"*{file_ext}")))

    # Load model with ort
    print("Loading onnx model..")
    session = ort.InferenceSession(checkpoint_fp)
    output_nodes = session.get_outputs()

    filename_list = []
    qs_scores_arr = np.zeros(len(img_list))
    for idx in tqdm(range(len(img_list))):
        img_p = img_list[idx]
        filename = img_p.relative_to(data_dir)
        filename_list.append(filename)

        # Prepare model input
        img = Image.open(img_p)
        tfms = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        img: torch.Tensor = tfms(img)
        img = img.unsqueeze(0).numpy()

        # Run inference
        outputs = session.run(
            output_names=["qs"],
            input_feed={"input": img},
        )
        # print(outputs[0].item())

        qs_scores_arr[idx] = outputs[0].item()

        if idx == 199:
            break

    print("Saving results..")
    qs_scores_arr = qs_scores_arr[: idx + 1]
    print("Shape of qs_scores_arr:", qs_scores_arr.shape)
    filename_arr = np.array(filename_list)

    df_out = pd.DataFrame({"sample": filename_arr, "quality": qs_scores_arr})
    print(df_out.head())
    df_out.to_csv(
        save_dir / f"{method_name}_{data_name}_onnx.txt",
        float_format="%.6f",
        header=True,
        index=False,
        sep=" ",
    )


if __name__ == "__main__":
    main()
