import os
import os.path as osp
import sys
import configparser
from terminaltables import AsciiTable
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm  # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from segmentation.dataset import SVDataset
import segmentation.transforms as T


def search_patient(loader, patient):
    for data in loader:
        if data['patient'][0] == patient:
            return data
    return None


def train(data, model, optimizer, losses, weights):
    model.train()
    epoch_total_loss = []

    img = data['img'].permute(4, 0, 1, 2, 3).to(device)  # NT, CH, NZ, NY, NX
    # img = data['img'].permute(4, 0, 3, 2, 1).to(device)  # NT, CH, NX, NY, NZ

    for t in range(img.shape[0] - 1):
        moving = img[t, ...].unsqueeze(0)
        fixed = img[t + 1, ...].unsqueeze(0)
        y_pred = model(moving, fixed)

        # calculate total loss
        y_true = [fixed, None]
        loss = 0
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss += curr_loss
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.array(epoch_total_loss).mean()


if __name__ == '__main__':
    # Seeding for reproducible results
    plots.seeding(42)

    # Read configuration options
    config = configparser.ConfigParser()
    config.read('parser/vxm_instance_train.ini')
    root_dir = config.get('DATA', 'ROOT_DIR')
    output_dir = config.get('DATA', 'OUTPUT_DIR')
    img_sz = config.getint('PARAMETERS', 'IMG_SIZE')
    workers = config.getint('PARAMETERS', 'WORKERS')
    epochs = config.getint('PARAMETERS', 'NUM_EPOCHS')
    bidir = config.getboolean('PARAMETERS', 'BIDIR')
    img_loss = config.get('PARAMETERS', 'IMG_LOSS')
    lambda_param = config.getfloat('PARAMETERS', 'LAMBDA')
    lr = config.getfloat('PARAMETERS', 'LR')
    cudnn_nondet = config.getboolean('PARAMETERS', 'CUDNN_NONDET')
    patient = config.get('DATA', 'PATIENT')
    model_weights = config.get('DATA', 'MODEL')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not cudnn_nondet
    print('cudnn.deterministic: ', (not cudnn_nondet))
    print('Weights: ', model_weights)

    save_dir = plots.createSaveDirectory(output_dir, 'VXM_IT')
    plots.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)

    # Create dataloaders
    val_transforms = T.Compose([
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        T.MinMaxNormalization(p=1.0),
        T.BinarizeMasks(th=0.5),
        T.ToTensor(add_ch_dim=False)
    ])

    test_ds = SVDataset(root_dir, 'test', val_transforms, vxm=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=workers, collate_fn=SVDataset.collate_fn)

    # load and set up model
    model = vxm.networks.VxmDense.load(model_weights, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if img_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif img_loss == 'mse':
        # image_loss_func = vxm.losses.MSE().loss
        image_loss_func = nn.MSELoss(reduction='sum')
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % img_loss)

    losses = [image_loss_func, vxm.losses.Grad('l2', loss_mult=2).loss]
    weights = [1.0, lambda_param]

    data = search_patient(test_loader, patient)
    if data is None:
        print('Patient: {} not found!'.format(patient))
        sys.exit()

    for e in tqdm(range(1, epochs + 1)):
        train_loss = train(data, model, optimizer, losses, weights)

        # Save model every 20 epochs
        if e % 10 == 0:
            model.save(os.path.join(save_dir, 'model.pth'))
        
        writer.add_scalar('loss', train_loss, e)

        print(AsciiTable([
            ['Split', 'Loss'],
            ['Train', '{:.6f}'.format(train_loss)]
        ]).table)

    model.save(os.path.join(save_dir, 'model.pth'))
