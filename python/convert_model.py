import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch.onnx
import torch.utils.model_zoo as model_zoo
from torch import nn

from cr_fiqa.backbones.iresnet import iresnet50 as crfiqa_iresnet50  # isort: skip
from cr_fiqa.backbones.iresnet import iresnet100 as crfiqa_iresnet100  # isort: skip


AVAIL_METHODS = [
    "crfiqa-s",
    "crfiqa-l",
]


def main(
    method_name: str = "crfiqa-l",
    checkpoint_fp: str = "checkpoints/pytorch/CRFIQA-L/181952backbone.pth",
    save_dir: str = "checkpoints/onnx",
):
    assert method_name in AVAIL_METHODS
    save_dir = Path(save_dir)  # type: Path
    if not save_dir.exists():
        save_dir.mkdir()

    # Load model
    print(f"Loading model for {method_name}..")
    if "crfiqa" in method_name:
        if method_name == "crfiqa-s":
            model = crfiqa_iresnet50(num_features=512, qs=1, use_se=False)
        if method_name == "crfiqa-l":
            model = crfiqa_iresnet100(num_features=512, qs=1, use_se=False)
        weight = torch.load(checkpoint_fp, weights_only=True)
        model.load_state_dict(weight)
    else:
        raise NotImplementedError()

    # Put model into eval mode
    model.eval()

    # Input to the model
    batch_size = 64
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
    feats, qs = model(x)

    print("feats shape:", feats.shape)
    print("qs shape:", qs.shape)

    # Export the model
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


if __name__ == "__main__":
    main()
