import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='checkpoint file')
    args = parser.parse_args()

    checkpoint = torch.load(args.file)

    print('Epoch: %d' % checkpoint['epoch'])
    print('\tTrain')
    print('\t\tLoss: %.3f' % checkpoint['train_loss'])
    print('\t\tAcc: %.3f' % checkpoint['train_acc'])
    print('\tVal')
    print('\t\tLoss: %.3f' % checkpoint['val_loss'])
    print('\t\tAcc: %.3f' % checkpoint['val_acc'])
    print('\tTest')
    print('\t\tAcc: %.3f' % checkpoint['test_acc'])
