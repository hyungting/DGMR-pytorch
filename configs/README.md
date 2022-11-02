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
