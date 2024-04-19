from yacs.config import CfgNode as CN

args = CN()

args.phase = 'train'

### log setting
args.save_dir = './ndbao/Sub_Pixel'
args.reset = True # Delete save_dir to create a new one
args.log_file_name = 'Sub_Pixel.log'
args.logger_name = 'Sub_Pixel'
args.writer_path = './ndbao/runs/Sub_Pixel'

### device setting
args.cpu = False
args.num_gpu = 1

### dataset setting
args.dataset = 'DIV2K'
args.dataroot_H = './vntan/Datasets/DIV2K/DIV2K_train_HR'
# args.dataroot_L = '/content/trainsets/trainL'
args.n_channels = 3
args.scale = 4
args.H_size = 192

args.data_eval_H = './ndbao/Data/testH'
args.data_eval_L = './ndbao/Data/testL'

### dataloader setting
args.num_workers = 2
args.batch_size = 8

### model setting
args.block_types = ['RB_A', 'RB', 'RRDB'] # [x1_type, x2_type, x4_type]
args.num_input_blocks = [8, 12, 1] # [x1_num_blocks, x2_num_blocks, x4_num_blocks]
args.up_type='pixelshuffle' # in: conv, pixelshuffle, pixelshuffle+conv, nearest+conv, bicubic+conv
args.num_sfe_blocks = 12
args.n_feats = 192
args.grow_channels = 32
args.dropout = 0.0
args.res_scale = 1.0
args.rgb_range = 1.0 # maximum value of RGB in Tensor

args.lte = CN()
args.lte.n_feats=64
args.lte.ch_mul=[1, 2, 4]
args.lte.attention_mul=[]
args.lte.dropout=0.0
args.lte.channels=3
args.lte.num_res_blocks=2

### loss setting
args.GAN_type = 'WGAN_GP'
args.GAN_k = 2
args.rec_type = 'l1'
args.rec_w = 1.
args.per_w = 0.
args.adv_w = 0.
args.D_path = None

### optimizer setting
args.beta1 = 0.9
args.beta2 = 0.999
args.eps = 1e-8
args.lr_rate = 1e-4
args.lr_rate_dis = 1e-4
args.lr_rate_lte = 1e-4
args.decay = [40, 80, 120]
args.gamma = 0.4

### training setting
args.train_crop_size = 48
args.num_init_epochs = 2
args.num_epochs = 999 # keep training
args.print_every = 10
args.val_every = 4
args.save_every = 4
args.save_all = True

### evaluate / test / finetune setting
args.eval = False
args.eval_save_results = False
args.model_path = None
args.test = False
args.lr_path = './'
args.ref_path = './'