import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tqdm import tqdm


def main(
    data_dir: str = "data/lfw-deepfunneled",
    data_name: str = "lfw",
    file_ext: str = ".jpg",
    method_name: str = "crfiqa-l",
    checkpoint_fp: str = "checkpoints/onnx/crfiqa-l.onnx",
    save_dir: str = "results",
):
    # Check directories
    data_dir = Path(data_dir)  # type: Path
    save_dir = Path(save_dir)  # type: Path
    if not save_dir.exists():
        save_dir.mkdir()

    # Glob data_dir
    img_list = sorted(list(data_dir.rglob(f"*{file_ext}")))

    # Read model and set input_size
    print("Loading onnx model..")
    # TODO: Load model with ort
    # model = cv2.dnn.readNetFromONNX(checkpoint_fp)
    input_size = (112, 112)
    # output_layer_names = model.getUnconnectedOutLayersNames()

    filename_list = []
    qs_scores_arr = np.zeros(len(img_list))
    # for idx in tqdm(range(len(img_list))):
    idx = 0
    img_p = img_list[idx]
    filename = img_p.relative_to(data_dir)
    filename_list.append(filename)
    # Prepare model input
    img = Image.open(img_p)
    img = np.array(img)
    print(img.shape)
    # Convert (H,W,C) to (C,H,W)
    img = img.transpose((2, 0, 1))
    print(img.shape)

    # TODO: Perform inference with ort
    # Make blobFromImage equivalent to transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # blob = cv2.dnn.blobFromImage(
    #     img,
    #     scalefactor=1 / (127.5),
    #     size=input_size,
    #     mean=(127.5, 127.5, 127.5),
    #     swapRB=True,
    #     crop=False,
    # )
    # # Run forward pass
    # model.setInput(blob)
    # feats, qs = model.forward(output_layer_names)
    # qs_scores_arr[idx] = qs.item()

    # # if idx == 199:
    # #     break

    # print("Saving results..")
    # qs_scores_arr = qs_scores_arr[: idx + 1]
    # print("Shape of qs_scores_arr:", qs_scores_arr.shape)
    # filename_arr = np.array(filename_list)

    # df_out = pd.DataFrame({"sample": filename_arr, "quality": qs_scores_arr})
    # print(df_out.head())
    # df_out.to_csv(
    #     save_dir / f"{method_name}_{data_name}_onnx.txt",
    #     float_format="%.6f",
    #     header=True,
    #     index=False,
    #     sep=" ",
    # )


if __name__ == "__main__":
    main()
