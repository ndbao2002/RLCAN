from yacs.config import CfgNode as CN

args = CN()

args.phase = 'train'

### log setting
args.save_dir = './ndbao/RNAN'
args.reset = True # Delete save_dir to create a new one
args.log_file_name = 'RNAN.log'
args.logger_name = 'RNAN'
args.writer_path = './ndbao/runs/RNAN'

### device setting
args.cpu = False
args.num_gpu = 1

### dataset setting
args.dataset = 'DIV2K'
args.dataroot_H = './Dataset/TrainDataset/DIV2K/DIV2K_train_HR'
# args.dataroot_L = '/content/trainsets/trainL'
args.n_channels = 3
args.scale = 2
args.L_size = 48

args.data_eval_H = './ndbao/Data/testH_x2'
args.data_eval_L = './ndbao/Data/testL_x2'

### dataloader setting
args.num_workers = 2
args.batch_size = 16

### model setting
args.scale = 2
args.n_resgroups = 6
args.n_resblocks = 6
args.n_feats = 180
args.heads = 4
args.window_size = 7

args.n_colors = 3
args.res_scale = 1.0
args.rgb_range = 1.0

args.pre_trained = None

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
args.decay = [4000, 6000, 7000, 7500]
args.gamma = 0.5

### training setting
args.train_crop_size = 48
args.num_init_epochs = 2
args.num_epochs = 9999 # keep training
args.print_every = 20
args.val_every = 5
args.save_every = 5
args.save_all = True