from easydict import EasyDict as edict
import time


# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 2023 # random seed,  for reproduction
__C.DATASET = 'SHHA' 

__C.GPU_ID = [0,1] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-5 # learning rate
__C.LR_DECAY = 0.995 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 200

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-4# SANet:0.001 CMTL 0.0001


# print 
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)


__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes

#------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes

#================================================================================
# init
__C_SHHA = edict()

cfg_SHHA = __C_SHHA

__C_SHHA.STD_SIZE = (768,1024)
__C_SHHA.TRAIN_SIZE = (576,768) # 2D tuple or 1D scalar
__C_SHHA.DATA_PATH = './datasets/ProcessedData/shanghaitech_part_A'               

__C_SHHA.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

__C_SHHA.LABEL_FACTOR = 1
__C_SHHA.LOG_PARA = 100.

__C_SHHA.RESUME_MODEL = ''#model path
__C_SHHA.TRAIN_BATCH_SIZE = 1 #must be 1
__C_SHHA.VAL_BATCH_SIZE = 1 # must be 1

#================================================================================
# init
__C_SHHB = edict()

cfg_SHHB = __C_SHHB

__C_SHHB.STD_SIZE = (768,1024)
__C_SHHB.TRAIN_SIZE = (576,768)
__C_SHHB.DATA_PATH = './datasets/ProcessedData/shanghaitech_part_B'               

__C_SHHB.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 100.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 6 #imgs

__C_SHHB.VAL_BATCH_SIZE = 6 # 

