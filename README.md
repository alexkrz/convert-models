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

The steps to download and convert the Pytorch checkpoints are described in `python/README.md`.

## C++ setup

The project requires `cmake` >= 3.23 and `conan` > 2.0.
If you cannot install these depencencies directly on your system, we recommend to use a conda virtual environment.

To set up the conda virtual environment, navigate into the `cpp` folder and run

```bash
conda env create -n conan -f environment.yml
```

The required packages are then installed in a conda environment with the name `conan`.

Fur further information on compiling the C++ code, look into `cpp/README.md`.

## Resources

List of useful resources:

- <https://github.com/onnx/tutorials>
- <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>

## Todos

- [x] Use opencv instead of onnxruntime
- [x] Compare results between pytorch model and onnx model
- [x] Add cpp implementation for onnx model
- [ ] Replace absolut paths in example.cpp
