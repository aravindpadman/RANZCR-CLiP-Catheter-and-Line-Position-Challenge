from enum import Enum


class ModelState(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    END = "end"


# class TrainingState(Enum):
#     TRAIN_START = "on_train_start"
#     TRAIN_END = "on_train_end"
#     EPOCH_START = "on_epoch_start"
#     EPOCH_END = "on_epoch_end"
#     TRAIN_EPOCH_START = "on_train_epoch_start"
#     TRAIN_EPOCH_END = "on_train_epoch_end"
#     VALID_EPOCH_START = "on_valid_epoch_start"
#     VALID_EPOCH_END = "on_valid_epoch_end"
#     TRAIN_STEP_START = "on_train_step_start"
#     TRAIN_STEP_END = "on_train_step_end"
#     VALID_STEP_START = "on_valid_step_start"
#     VALID_STEP_END = "on_valid_step_end"
#     TEST_STEP_START = "on_test_step_start"
#     TEST_STEP_END = "on_test_step_end"



class TrainingState(Enum):
    EPOCH_BEGIN = "on_epoch_begin"
    EPOCH_END = "on_epoch_end"
    TRAIN_BATCH_BEGIN = "on_train_batch_begin"
    TRAIN_BATCH_END = "on_train_batch_end"
    VALIDATION_BATCH_BEGIN = "on_validation_batch_begin"
    VALIDATION_BATCH_END = "on_validation_batch_end"
    TEST_BATCH_BEGIN = "on_test_batch_begin"
    TEST_BATCH_END = "on_test_batch_end"
    TRAIN_BEGIN = "on_train_begin"
    TRAIN_END = "on_train_end"
    VALIDATION_BEGIN = "on_validation_begin"
    VALIDATION_END = "on_validation_end"
    TEST_BEGIN = "on_test_begin"
    TEST_END = "on_test_end"
