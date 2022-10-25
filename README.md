# DGMR-pytorch
An implementation of Deep Generative Model of Radars from DeepMind in PyTorch

## Dependencies

- dask==2022.9.1
- matplotlib==3.5.1
- numba==0.55.1
- numpy==1.21.0
- pandas==1.4.1
- properscoring==0.1
- pytorch-lightning==1.5.10
- torchvision==0.11.3
- pytorch

Execute ```pip install -r requirements.txt``` to install required packages (except for torch).

For installing torch, please check out https://pytorch.org/get-started/locally/ to figure out the version that works on your device.

## Workflow
<img width="860" alt="image" src="https://user-images.githubusercontent.com/51833239/197740656-91621804-2096-47cd-9f3b-1547cb45d066.png">


## Data preparation
- **Step 1**: Read your own data.
- **Step 2**: Store your data into ```numpy.ndarray``` with proper data type (e.g. ```int16```, ```float64```).
- **Step 3**: Make sure your data are sorted by time.
- **Step 4**: Note the array's storing *data type* and *data shape*, these information will be needed in config file.
- **Step 5**: Save the array with the format ```.dat``` or ```.npy```. (No headers please!)

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
- **Step 1**: Prepare data (see **Data Preparation**).
- **Step 2**: Prepare rain records ```.csv``` file. (optional)
If rain records are not prepared, it will be calculated automatically in our program. 
Example of csv:    

| index | nonzeros                        |
| ----- | ------------------------------- |
| 0     | number of nonzeros of image 0   |
| 1     | number of nonzeros of image 1   |
| ...   | number of nonzeros of images    |
| N-1   | number of nonzeros of image N-1 |

- **Step 3**: Prepare config file.
Please see configs/README.md 

- **Step 4**: Execute the code
    - train: ```python3 main.py -c /path/to/your/config -m train```
    - validate: ```python3 main.py -c /path/to/your/config -m val```
    - test: ```python3 main.py -c /path/to/your/config -m test```
    
## Tensorboard
Execute this command to see training results.
```tensorboard --logdir /path/where/records/are/stored```
