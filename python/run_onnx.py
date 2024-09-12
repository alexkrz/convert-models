import onnx
import onnxruntime
import torch


def main(model_p: str = "checkpoints/onnx/crfiqa-l.onnx"):
    batch_size = 64
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)

    onnx_model = onnx.load(model_p)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        model_p,
        providers=["CPUExecutionProvider"],
    )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    print(ort_outs[0].shape)


if __name__ == "__main__":
    main()
