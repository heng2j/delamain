from lane_tracking.data.constant import culane_row_anchor


# DATA
model_name = 'lane_tracking/culane_18.pth'
dataset = 'CULane'
data_root = 'data/testdata/train/'

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = None

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4
cls_num_per_lane = 18
row_anchor = culane_row_anchor
