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
Please see **Parameters of config**

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



## Parameters of config

#### **PARAMS**
- **BATCH_SIZE**: int
- **EPOCH**: int
- **N_SAMPLE**: int, the number of generated samples per input of training generator.
- **INPUT_FRAME**: int, the number of rainfall frames that are used to generated future frames.
- **OUTPUT_FRAME**: int, the number of generated future frames.
- **COVERAGE**: float, the acceptable precipitation of a single rainfall frame.
- **CROP_SIZE**: int, the desired size of input and output frames.
- **PARSER**
    - **FUNCTION**: str, the name of function for parsing data.
    - **PARAMS**
        - **ARGS**: arguments of parser function.
- **NORMALIZER**
    - **FUNCTION**: str, the name of function for normalizing data.
    - **PARAMS**
        - **ARGS**: arguments of normalizer function.

#### **GENERATOR**
- **MODEL**
    - **FUNCTION**: str, the name of generator function.
    - **PARAMS**
        - **ARGS**: arguments of model function.
- **LOSS**
    - **FUNCTION**: str, the name of loss function used for training generator.
    - **PARAMS**
        - **ARGS**: arguments of loss function.
- **OPTIMIZER**
    - **FUNCTION**: str, the name of optimizer for generator.
    - **PARAMS**
        - **ARGS**: arguments of optimizer function.
- **SCHEDULER**
    - **FUNCTION**: str, the name of learning rate scheduler.
    - **PARAMS**
        - **ARGS**: arguments of scheduler function.

#### **DISCRIMINATOR**
- ITER: int, the number of iterations of training D for one epoch.
- **MODEL**
    - **FUNCTION**: str, the name of generator function.
    - **PARAMS**
        - **ARGS**: arguments of model function.
- **LOSS**
    - **FUNCTION**: str, the name of loss function used for training discriminator.
    - **PARAMS**
        - **ARGS**: arguments of loss function.
- **OPTIMIZER**
    - **FUNCTION**: str, the name of optimizer for discriminator.
    - **PARAMS**
        - **ARGS**: arguments of optimizer function.
- **SCHEDULER**
    - **FUNCTION**: str, the name of learning rate scheduler.
    - **PARAMS**
        - **ARGS**: arguments of scheduler function.

#### **EVALUATION**
- **THRESHOLDS**: list of floats, The rainfall thresholds for evaluating prediction's forecast skill.
- **POOLING_SCALES**: list of ints, the aggregation scale for calculating Continuous Ranked Probability Score.

#### **TRAIN**
- **VAL_PERIOD**: int, perform validation every N training epochs.
- **CHECKPOINT_PERIOD**: int, save checkpoint every N training epochs.

#### **VAL**
- **BATCH_SIZE**: int

#### **TEST**
- **BATCH_SIZE**: int

#### **DATALOADER**
- **NUM_WORKERS**: int
- **PIN_MEMORY**: int

#### **SETTINGS**
- **IMPORT_CKPT_PATH**: str, path of checkpoint that are expected to be loaded.
- **NUM_GPUS**: int
- **RNG_SEED**: int
- **DATA_PATH**: str or list of strs, path of dataset (npy file only).
- **DATA_TYPE**: str or list of strs, storing data type.
- **DATA_SHAPE**: list or list of lists, shape of data path.
- **RAIN_RECORD_PATH**: str or list of strs, path of statistics of dataset.

#### **TENSORBOARD**
- **SAVE_DIR**: str, path of storing training records
- **VERSION**: str, unique name of current training record
