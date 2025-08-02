# APU-TrajGen+

This is the code for generating the results presented in the article: ***APU-TrajGen+: GRU-based Adaptive Privacy and
Utility Preserving Trajectory Generation***

## Environment Requirements

- Python >= 3.11.5
- The requirements.txt file is included

## Datasets

- Porto - https://doi.org/10.24432/C55W25
- San Francisco - https://ieee-dataport.org/open-access/crawdad-epflmobility

## Reproducibility & Run

### Model generation

=> "training" folder

### Trajectory generation

=> "trajgen" folder

**Step 1**. Run the [trajgen-fixed-k notebook](trajgen/trajgen-fixed-k.ipynb) for the targeted dataset.

**Step 2**. Extract the values of MDE_k for various values of k (k >=1). MDE_k values are necessary for setting up the adaptive k approach.

**Step 3**. Run the trajgen-adaptive-k.ipynb notebook for generating the trajectories using the adaptive k approach.

### Use case

...

## Acknowledgement

Any scientific publications that use our data or code should mention the article and/or this git.
