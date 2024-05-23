# Single Ventricle Segmentation

## Installation

1. Create a conda environment using the provided `environment.yaml` file:
    ```bash
    conda env create -f environment.yaml
    ```

## Instructions for the ACDC Dataset

The ACDC dataset contains information for 3 ROIs (left ventricle, right ventricle, and myocardium). These ROIs must be separately extracted and preprocessed before computing the optical flow and performing CNN postprocessing.

Configuration parameters can be found in `conf/setup_acdc.yaml` and `conf/preprocessing.yaml`. There are two different ways to preprocess the ACDC dataset:

### Method 1: All-in-One

This method reads the raw ACDC dataset folder and runs all preprocessing steps listed in `scripts/run_acdc_preprocessing.py`. A folder for each preprocessing step will be created in the `out_dir` directory, which is specified in the `conf/preprocessing.yaml` configuration file.

```bash
python scripts/run_acdc_preprocessing.py
```

### Method 2: Step-by-Step

You can also preprocess the ACDC dataset by independently running each preprocessing script. Adjust settings as needed by editing `conf/preprocessing.yaml`.

1. **Reformat the ACDC Dataset**: Set up the directory structure and filename convention used in this project.

    ```bash
    python svs/preprocessing/setup_acdc.py
    ```

    Use the `conf/setup_acdc.yaml` file to set the correct paths and desired configuration.

2. **Cutting**: Extract relevant portions of the dataset.

    ```bash
    python svs/preprocessing/cutting.py
    ```

3. **Prolongation**: Adjust the dimensions as needed.

    ```bash
    python svs/preprocessing/prolongation.py
    ```

4. **Normalization**: Normalize the intensity values in the dataset.

    ```bash
    python svs/preprocessing/normalization.py
    ```

5. **Split**: Split the dataset for training, validation, and testing.

    ```bash
    python svs/preprocessing/split.py
    ```

## Optical Flow

This repository provides two methods for computing 3D optical flow using the TVL1-3D formulation: 
1. Directly running the `scripts/flow.py` Python script.
2. Using the `scripts/parallel_flow.sh` bash script for parallel processing with multiple GPUs.

### Method 1: Using `scripts/flow.py`
This method involves directly running the Python script to compute optical flow. This approach is straightforward and suitable for single or sequential processing.
1. First, ensure the configuration file `conf/flow.yaml` is correctly set up with the necessary parameters.

2. Then, run the script:
    ```bash
    python scripts/flow.py
    ```

### Method 2: Using `scripts/parallel_flow.sh`
This method involves using a bash script to run the optical flow computation in parallel on multiple GPUs. This approach maximizes the use of available computational resources, allowing for faster processing.

1. **Make the Script Executable**:
   ```bash
   chmod +x scripts/parallel_flow.sh
   ```

2. **Run the Script**:
   ```bash
   ./scripts/parallel_flow.sh
   ```

   - The script will automatically create a `results` directory if it doesn't exist.
   - It will start the forward and backward optical flow computations in parallel on different GPUs, logging the output to separate files.
   - You can monitor the progress of each computation by checking the log files in the `results` directory. For example:
        ```bash
        tail -f results/fwd_0-25.log
        ```
        This command shows the last few lines of the log file and updates as new lines are added.

## Warping optical flow
The optical flow algorithm's performance can be assessed by propagating the ground truth masks at ED and ES cardiac phases. This can be done by running `scripts/warp.py`. The configuration parameters can be set using `conf/warping.yaml`. The results should like this animation.
![Alt text](docs/animation.gif)

## Training the CNN

*Instructions for training the CNN will go here.*

## Notes

- Ensure that you have the necessary permissions to read and write to the specified directories.
- Verify that all required dependencies are installed in your conda environment.
- Adjust the configuration files as needed for your specific dataset and processing requirements.
