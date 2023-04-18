import torch
from torch.utils.data import DataLoader
import sys
import os.path as osp
from cnn.experimental.acdc_dataset import *
import configparser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from monai.metrics.meandice import compute_meandice
from monai.losses.dice import DiceCELoss, DiceLoss
from utilities import cnn_utils

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import plots
import utilities.binary_transforms as T


def plot_pred(writer, pred, mask, tag, p=0.2):
    if np.random.rand() < p:
        # save_dir = plots.createSubDirectory(save_dir, tag)
        # plots.save_img_masks(img.squeeze(), [mask.squeeze(), pred.squeeze()], filename, save_dir,
        #                      th=0.5, alphas=[0.4, 0.4], colors=[[1, 0.1, 0.2], [0.1, 1.0, 0.3]])v       

        b = np.random.randint(mask.shape[0])
        pred = pred[b]
        mask = mask[b]
        pred = pred.swapaxes_(0, 1)
        mask = mask.swapaxes_(0, 1)
        error = torch.abs(mask - pred)
        writer.add_images(f'{tag}/gt', mask)
        writer.add_images(f'{tag}/est', pred)
        writer.add_images(f'{tag}/error', error)


def train(net, opt, loss_fn, train_loader, writer, report):
    net.train()
    for data in train_loader:
        img, mask = data
        img = img.to(device)
        mask = mask.to(device)

        pred, _ = net(img)
        loss = loss_fn(pred, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            report['train_loss'] += loss.item()
            pred = torch.where(pred > 0.5, 1.0, 0.0)
            report['train_acc'] += compute_meandice(pred, mask).mean().item()
            plot_pred(writer, pred, mask, 'train')
    
    report['train_loss'] = report['train_loss'] / len(train_loader)
    report['train_acc'] = report['train_acc']  / len(train_loader)
    return report

@torch.no_grad()
def val(net, loss_fn, val_loader, writer, report):
    net.eval()
    for data in val_loader:
        img, mask = data
        img = img.to(device)
        mask = mask.to(device)

        pred, _ = net(img)
        loss = loss_fn(pred, mask)

        report['val_loss'] += loss.item()
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        report['val_acc'] += compute_meandice(pred, mask).mean().item()
        plot_pred(writer, pred, mask, 'val')    
    
    report['val_loss'] = report['val_loss'] / len(val_loader)
    report['val_acc'] = report['val_acc'] / len(val_loader)
    return report 


def create_checkpoint(e, net, opt, report, save_dir, filename):
    torch.save({
        'epoch': e,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_loss': report['train_loss'],
        'train_acc': report['train_acc'],
        'val_loss': report['val_loss'],
        'val_acc': report['val_acc'],
        'test_acc': 0
    }, osp.join(save_dir, filename))


def log(net, opt, e, report, best_report, writer):
    if report['train_acc'] > best_report['train_acc']:
        best_report['train_acc'] = report['train_acc']
        create_checkpoint(e, net, opt, report, save_dir, 'best_train_checkpoint.pth')
    
    if report['val_acc'] > best_report['val_acc']:
        best_report['val_acc'] = report['val_acc']
        create_checkpoint(e, net, opt, report, save_dir, 'best_val_checkpoint.pth')
        
    writer.add_scalar('lr', opt.param_groups[0]['lr'], e)
    writer.add_scalars('loss', {'train': report['train_loss'], 'val': report['val_loss']}, e)
    writer.add_scalars('acc', {'train': report['train_acc'], 'val': report['val_acc']}, e)
    return best_report


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = cnn_utils.read_train_params(config)

    # Create dataset and data loaders
    train_transforms = T.Compose([
        T.ToTensor(),
        T.RandomVerticalFlip(P['vflip_prob']),
        T.RandomHorizontalFlip(P['hflip_prob']),
        T.RandomDepthFlip(P['dflip_prob']),
        T.RandomRotate(P['rot_prob'], P['rot_range_x'], P['rot_range_y'], P['rot_range_z'], P['rot_boundary'], P['clip_interval']),
        T.ElasticDeformation(P['ed_prob'], P['ed_sigma_range'], P['ed_grid'], P['ed_boundary'], P['ed_prefilter'], P['ed_axis'], P['clip_interval']),
        T.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
        T.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
        T.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
        T.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval']),
        T.BinarizeMasks(th=0.5)])

    val_transforms = T.Compose([
        T.ToTensor(),
        T.BinarizeMasks(0.5)])
   
    timepoint = Timepoint.ED

    train_ds = ACDCDataset(config, DatasetMode.TRAIN, timepoint, train_transforms)
    val_ds = ACDCDataset(config, DatasetMode.VAL, timepoint, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=True, num_workers=P['workers'])
    val_loader = DataLoader(val_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['workers'])

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)

    net = cnn_utils.create_net(config, logger).to(device)
    summary(net, input_size=(1, 80, 80, 80), batch_size=P['batch_size'])
    net = torch.nn.DataParallel(net, device_ids=np.arange(P['gpus']).tolist())
    if P['pretrained']:
        checkpoint_file = P['checkpoint_file']
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f'Use pretrained weights: {checkpoint_file}')
        opt = optim.Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        opt = optim.Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
    
    plots.save_config(config,save_dir, 'config.ini')
    cnn_utils.save_model(net, save_dir, 'net.txt')
    train_ds.save_patients(save_dir, 'train.xlsx')
    val_ds.save_patients(save_dir, 'val.xlsx')

    opt = optim.Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=P['step_size'], gamma=P['gamma'])

    # loss_fn = nn.BCELoss()
    loss_fn = DiceCELoss(lambda_ce=0.4, lambda_dice=0.6)
    # loss_fn = DiceLoss()

    logger.info('Save dir: %s' % save_dir)
    logger.info('Training CNN')
    best_report = {'train_acc': 0, 'val_acc': 0}
    pbar = tqdm(total=P['epochs'])

    for e in range(1, P['epochs']):
        report = {'train_loss': 0, 'val_loss': 0, 'train_acc': 0, 'val_acc': 0}
        
        report = train(net, opt, loss_fn, train_loader, writer, report)
        report = val(net, loss_fn, val_loader, writer, report)
        
        best_report = log(net, opt, e, report, best_report, writer)
        scheduler.step()
        pbar.update(1)
    
    create_checkpoint(e, net, opt, report, save_dir, 'checkpoint.pth')
    