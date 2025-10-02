from collections import OrderedDict

import torch

from .cr_fiqa.backbones.iresnet import iresnet50 as crfiqa_iresnet50
from .cr_fiqa.backbones.iresnet import iresnet100 as crfiqa_iresnet100
from .debfiqa.debfiqa import DebiasedFIQA
from .debfiqa.iresnet_magface import iresnet100 as debfiqa_iresnet100


def adjust_debfiqa_dict(model: torch.nn.Module, state_dict: OrderedDict) -> OrderedDict:
    adjusted_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("backbone.features.", "backbone.")
        if new_k in model.state_dict().keys() and v.size() == model.state_dict()[new_k].size():
            adjusted_dict[new_k] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(adjusted_dict.keys())
    # assert num_model == num_ckpt, "Sizes of model keys and checkpoint keys do not match"
    return adjusted_dict


def load_model(method_name: str, checkpoint_fp: str):
    # Load model
    print(f"Loading model for {method_name}..")
    if "crfiqa" in method_name:
        if method_name == "crfiqa-s":
            model = crfiqa_iresnet50(num_features=512, qs=1, use_se=False)
        if method_name == "crfiqa-l":
            model = crfiqa_iresnet100(num_features=512, qs=1, use_se=False)
        weight = torch.load(checkpoint_fp, weights_only=True)
        model.load_state_dict(weight)
    elif "debfiqa" in method_name:
        weight = torch.load(checkpoint_fp, weights_only=True)
        # print(weight.keys())
        backbone = debfiqa_iresnet100()
        backbone.eval()
        place_holder = torch.zeros([3, 112, 112]).unsqueeze(0)
        feature_dims = backbone.get_feature_sizes(place_holder)
        model = DebiasedFIQA(backbone=backbone, feature_dims=feature_dims)
        weight = adjust_debfiqa_dict(model, weight)
        model.load_state_dict(weight, strict=False)
    else:
        raise NotImplementedError()

    return model
