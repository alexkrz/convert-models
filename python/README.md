# Convert model checkpoints to ONNX

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

Now, we can convert the model by activating the conda environment and running from the root directory:

```bash
conda activate convert
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

To test the onnx model in a CPP environment, we need to setup the CPP environment as described in `cpp/README.md`.
