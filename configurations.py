# Configurations
IMG_SIZE = 32
CLASSES = ['calling', 'distracted', 'drinking', 'looking_backward', 'not_distracted', 'texting']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)