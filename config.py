#ENCODER = 'se_resnext50_32x4d'
# ENCODER = 'efficientnet-b7'
ENCODER = 'efficientnet-b5'
#ENCODER = 'densenet161'
#ENCODER = 'densenet201'
#ENCODER =
ENCODER_WEIGHTS = 'imagenet'
#CLASSES = ['IMA']
CLASSES = ['NA']
#CLASSES = ['NE']
#CLASSES = ['NE_nohealth']
ACTIVATION = 'sigmoid'  
DEVICE = 'cuda'
NAME = 'UNetpp_DICE'

# DATA_DIR = './data/A. Segmentation_IMA/'
DATA_DIR = './data/A. Segmentation_' + CLASSES[0] + '/'
