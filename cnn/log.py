import torch
import os.path as osp

__all__ = ['checkpoint', 'update_train_history', 'log']


def checkpoint(e, net, opt, train_res, val_res, test_acc_avg, save_dir, filename):
    torch.save({
        'epoch': e,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_loss': train_res[0],
        'train_acc': train_res[-1],
        'val_loss': val_res[0],
        'val_acc': val_res[-1],
        'test_acc': test_acc_avg
    }, osp.join(save_dir, filename))


def update_train_history(H, train_avg, val_avg, test_acc_avg):
    H['train_loss'].append(train_avg[0])
    H['val_loss'].append(val_avg[0])
    H['train_acc'].append(train_avg[-1])
    H['val_acc'].append(val_avg[-1])
    H['test_acc'].append(test_acc_avg)
    return H


def log(logger, writer, e, train_avg, val_avg, test_acc_avg, net, opt, best_train_loss, best_val_loss, save_dir):
    log_train = '\t*Train:'
    log_val = '\t*Val:'
    log_test = f'\t*Test:\tacc: {test_acc_avg:,.3f}'
    for i in range(len(train_avg) - 1):
        writer.add_scalars(f'loss/l{i}', {'train': train_avg[i], 'val': val_avg[i]}, e)
        log_train += f'\tl{i}: {train_avg[i]:,.3f}'
        log_val += f'\tl{i}: {val_avg[i]:,.3f}'
    log_train += f'\tacc: {train_avg[-1]:,.3f}'
    log_val += f'\tacc: {val_avg[-1]:,.3f}'
    logger.info(f'Epoch: {e}')
    logger.info(log_train)
    logger.info(log_val)
    logger.info(log_test)

    writer.add_scalar('lr', opt.param_groups[0]['lr'], e)
    writer.add_scalars('acc', {'train': train_avg[-1], 'val': val_avg[-1], 'test': test_acc_avg}, e)

    if train_avg[0] < best_train_loss:
        best_train_loss = train_avg[0]
        checkpoint(e, net, opt, train_avg, val_avg, test_acc_avg, save_dir, 'best_train_checkpoint.pth')
        logger.info(f'\t*Best train checkpoint updated with loss: {best_train_loss:,.3f}')
    if val_avg[0] < best_val_loss:
        best_val_loss = val_avg[0]
        checkpoint(e, net, opt, train_avg, val_avg, test_acc_avg, save_dir, 'best_val_checkpoint.pth')
        logger.info(f'\t*Best val checkpoint updated with loss: {best_val_loss:,.3f}')

    return best_train_loss, best_val_loss
