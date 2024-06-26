from argument.argument_dnan import args
from utils.utils import mkExpDir, save_model, calc_psnr_and_ssim_torch_metric
from dataset.dataset import Train_Dataset, Test_Dataset
from network.DNAN import DNAN
from network.utils import Vgg19
from loss import get_loss_dict
import os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torch import optim
import torchvision
from torch.nn import functional as F



if __name__ == '__main__':
    # Setup logger
    logger = mkExpDir(args)
    logger.info('LOGGER SETUP COMPLETED')
    
    # Define dataset
    train_dataset = Train_Dataset(image_path=args.dataroot_H, 
                                patch_size=args.L_size, 
                                scale=args.scale)

    train_dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,)
    
    # Eval dataset

    eval_dataset = Test_Dataset(image_path_HR=args.data_eval_H, 
                                image_path_LR=args.data_eval_L)

    eval_dataloader = DataLoader(eval_dataset,
                            batch_size=1,
                            num_workers=1)
    
    logger.info('DATASET LOADED')
    
    # Define model
    model = DNAN(args)
    
    if args.pre_trained:
        model.load_state_dict(
            torch.load(args.pre_trained),
            strict=False,)
    logger.info('MODEL LOADED')
    
    # Optimizer, lr scheduler
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    logger.info(f'Device: {device}')

    # vgg19 = Vgg19(requires_grad=False).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.decay,
        gamma=args.gamma
    )
    
    # Define loss function
    loss_all = get_loss_dict(args, logger)
    
    # Summary Writer
    writer = SummaryWriter(log_dir=args.writer_path)
    
    # Train
    model.to(device)

    last_trained_path = "/media/btlen03/ndbao/pre_trained_model/DNAN/current_model.pth"
    save_all_training = True
    if last_trained_path:
        data = torch.load(os.path.join(last_trained_path))
        if save_all_training:
            # scheduler.load_state_dict(data['scheduler'])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=args.decay,
                gamma=args.gamma,
                last_epoch=data['epoch'],
            )
            optimizer.load_state_dict(data['opt'])
            
            lr = scheduler.get_last_lr()[0]
            
            for g in optimizer.param_groups:
                g['lr'] = lr
            
        model.load_state_dict(data['model'])
        logger.info(f'Load Pretrained model at epoch {data['epoch']}')
        count = data['step']
        start_epoch = data['epoch']
        log_loss = data['loss']
    else:
        count = 0
        start_epoch = 0
        log_loss = []
    max_psnr = 38.052
    max_psnr_epoch = 1135
    max_ssim = 0.9611
    max_ssim_epoch = 1220

    for epoch in range(start_epoch+1, args.num_epochs+1):
        model.train()
        for imgs in train_dataloader:
            hr, lr = imgs
            hr = hr.to(device)
            lr = lr.to(device)

            sr = model(lr)

            rec_loss = args.rec_w * loss_all['rec_loss'](sr, hr)
            loss = rec_loss

            # if epoch > args.num_init_epochs:
            #     if ('per_loss' in loss_all):
            #         sr_relu5_1 = vgg19((sr + 1.) / 2.)
            #         with torch.no_grad():
            #             hr_relu5_1 = vgg19((hr.detach() + 1.) / 2.)
            #         per_loss = args.per_w * loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
            #         loss += per_loss
            #     if ('adv_loss' in loss_all):
            #         adv_loss = args.adv_w * loss_all['adv_loss'](sr, hr)
            #         loss += adv_loss

            if count % args.print_every == 0:
                logger.info('Epoch {}/{}, Iter {}: Loss = {}, lr = {}'.format(
                    epoch,
                    args.num_epochs,
                    count,
                    loss.mean().item(),
                    scheduler.get_last_lr(),
                ))
                writer.flush()

            log_loss.append(loss.mean().item())
            writer.add_scalar('Loss/train', loss.mean().item(), count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

        scheduler.step()

        if epoch % args.val_every == 0:
            logger.info(f'Evaluate at epoch {epoch}!')
            model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0

                # img_eval_dir = os.path.join(args.save_dir, 'model_eval', f'{epoch}')
                # os.mkdir(img_eval_dir)

                for imgs in eval_dataloader:
                    cnt += 1
                    hr, lr = imgs
                    hr = hr.to(device)
                    lr = lr.to(device)
                    
                    h_old = lr.size(2)
                    w_old = lr.size(3)
                        
                    sr = model(lr)

                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim_torch_metric(sr.detach(), hr.detach())

                    lr = F.interpolate(lr, (hr.size(2), hr.size(3)), mode='bicubic')

                    # torchvision.utils.save_image(torch.concat([hr, sr, lr], dim=0),
                    #                             os.path.join(img_eval_dir, f'Set5_{cnt}.png'))

                    psnr += _psnr
                    ssim += _ssim
                psnr_avg = psnr / cnt
                ssim_avg = ssim / cnt
                if psnr_avg > max_psnr:
                    max_psnr = psnr_avg
                    max_psnr_epoch = epoch
                    save_model(args.save_all, model, optimizer, scheduler, count, epoch, log_loss,
                            os.path.join(args.save_dir, 'model', 'max_psnr_model.pth'))
                if ssim_avg > max_ssim:
                    max_ssim = ssim_avg
                    max_ssim_epoch = epoch
                    save_model(args.save_all, model, optimizer, scheduler, count, epoch, log_loss,
                            os.path.join(args.save_dir, 'model', 'max_ssim_model.pth'))
                logger.info('Eval  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                        %(max_psnr, max_psnr_epoch, max_ssim, max_ssim_epoch))
                logger.info('Eval  PSNR (current): %.3f (%d) \t SSIM (current): %.4f (%d)'
                        %(psnr_avg, epoch, ssim_avg, epoch))
            logger.info('Evaluation over.')
        if epoch % args.save_every == 0:
            save_model(args.save_all, model, optimizer, scheduler, count, epoch, log_loss,
                            os.path.join(args.save_dir, 'model', 'current_model.pth'))


