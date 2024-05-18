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

3. **Prolongation**: Adjust the time dimension as needed.

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
*Instructions for the optical flow will go here.*

<!-- After preprocessing the data, compute the optical flow:

```bash
python TVL1OF/compute_flow.py
```

This will create a folder named `optical_flow` inside the `{root_dir}`. Configure the paths and settings in `parser/flow_compute.ini`. -->

## Training the CNN

*Instructions for training the CNN will go here.*

## Notes

- Ensure that you have the necessary permissions to read and write to the specified directories.
- Verify that all required dependencies are installed in your conda environment.
- Adjust the configuration files as needed for your specific dataset and processing requirements.

This README should now provide a clear and concise guide for setting up, preprocessing the ACDC dataset, computing optical flow, and training the CNN for single ventricle segmentation.