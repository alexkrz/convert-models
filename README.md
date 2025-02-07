# Convert Models

The conversion is expected to be run from the root directory of the project.
First, we download the official Pytorch checkpoints from CR-FIQA: <https://github.com/fdbtrs/CR-FIQA>.

We put the checkpoints in a folder `checkpoints/` within the root directory.
Our folder structure should then look as follows:

```md
root/
├── checkpoints/
│   ├── onnx/
│   │   ├── ...
│   └── pytorch/
│       ├── CRFIQA-L/
│       │   ├── 181952backbone.pth
│       │   └── training.log
│       └── CRFIQA-S/
│           └── 32572backbone.pth
├── cpp/
├── data/
├── python/
├── results/
```

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

Now, we can convert the model by activating the conda environment and running from the root directory:

```bash
python python/convert_model.py
```

## Test the pytorch model on sample data

For testing the model checkpoints, we use the Labeled Faces in the Wild (LFW) dataset: <https://www.kaggle.com/datasets/jessicali9530/lfw-dataset>.

We unzip the images into the data directory at `data/lfw-deepfunneled/`.

Next, we can run the Pytorch script by running:

```bash
python python/compute_quality.py
```

The results will be stored in a `.txt` file at `results/crfiqa-l_lfw.txt`.

## Test the onnx model on sample data

Next, we want to test the converted onnx model on the LFW dataset.

Therefore, we run the following python script:

```bash
python python/run_onnx.py
```

The results will be stored in a `.txt` file at `results/crfiqa-l_lfw_onnx.txt`.

## Test the onnx model in CPP framework

To test the onnx model in a CPP framework, we need to setup the corresponding environment.

The project requires `cmake` >= 3.23 and `conan` > 2.0.
If you cannot install these dependencies directly on your system, we recommend to use a conda virtual environment.

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
