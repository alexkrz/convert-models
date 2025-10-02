import os
from pathlib import Path

import numpy as np
import torch.onnx
import torch.utils.model_zoo as model_zoo
from torch import nn

from src.utils import load_model  # isort: skip


def main(
    method_name: str = "debfiqaubmx_31",
    checkpoint_fp: str = "checkpoints/pytorch/DEBFIQAUBMX/debfiqaubmx_31.pth",
    save_dir: str = "checkpoints/onnx",
):
    save_dir = Path(save_dir)  # type: Path
    if not save_dir.exists():
        save_dir.mkdir()

    # Load model
    model = load_model(method_name, checkpoint_fp)
    model.eval()

    # Input to the model
    batch_size = 64
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)

    feats, qs = model(x)
    print("feats shape:", feats.shape)
    print("qs shape:", qs.shape)

    # Export the model
    print("Exporting model..")
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        f"{str(save_dir)}/{method_name}.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["feats", "qs"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "feats": {0: "batch_size"},
            "qs": {0: "batch_size"},
        },
    )

    print(f"Model successfully exported to {str(save_dir)}/{method_name}.onnx")


if __name__ == "__main__":
    main()
