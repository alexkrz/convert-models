# Convert Models

## Python setup

1. Navigate into `python` folder

2. Set up conda environment

    ```bash
    conda env create -n convert -f environment.yml
    ```

3. Install pip dependencies

    ```bash
    conda activate convert
    pip install -r requirements.txt
    ```

## Resources

List of useful resources:

- <https://github.com/onnx/tutorials>
- <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>

## Todos

- [x] Use opencv instead of onnxruntime
- [x] Compare results between pytorch model and onnx model
- [ ] Add cpp implementation for onnx model
