# DGMR-pytorch
An implementation of Deep Generative Model of Radars from DeepMind in PyTorch

## Dependencies

- python==3.8
- numpy==1.23.3
- pytorch_lightning==1.7.7
- torchvision==0.13.1
- pandas==1.5.0
- properscoring==0.1
- matplotlib==3.6.1
- dask==2022.9.2
- numba==0.56.2
- torch

Execute ```pip install -r requirements.txt``` to install required packages (except for torch).

For installing torch, please check out https://pytorch.org/get-started/locally/ to figure out the version that works on your device.

## Data preparation
**Step 1**: Read your own data.
**Step 2**: Store your data into ```numpy.ndarray``` with proper data type (e.g. ```int16```, ```float64```).
**Step 3**: Make sure your data are sorted by time.
**Step 4**: Note the array's storing *data type* and *data shape*, these information will be needed in config file.
**Step 5**: Save the array with the format ```.npy```.

For more information, please check out numpy's documentation.
- Data types
    https://numpy.org/doc/stable/user/basics.types.html
- How to save array?
    https://numpy.org/doc/stable/reference/generated/numpy.save.html
- Acceptable data shape: (N, H, W)
    - N: number of rainfall images
    - H: height of rainfall images
    - W: width of rainfall images

## How to run DGMR?

**Step 1**: Prepare data (see **Data Preparation**).
**Step 2**: Prepare rain records ```.csv``` file. (optional)
If rain records are not prepared, it will be calculated automatically in our program. 
Example of csv:    
| index | nonzeros                        |
| ----- | ------------------------------- |
| 0     | number of nonzeros of image 0   |
| 1     | number of nonzeros of image 1   |
| ...   | number of nonzeros of images    |
| N-1   | number of nonzeros of image N-1 |

**Step 3**: Prepare config file.
Please see configs/README.md 

**Step 4**: Execute the code
- train
    ```python3 main.py -c /path/to/your/config -m train```
- validate
    ```python3 main.py -c /path/to/your/config -m val```
- test
    ```python3 main.py -c /path/to/your/config -m test```
    
## Tensorboard
Execute this command to see training results.
```tensorboard --logdir /path/where/records/are/stored```
