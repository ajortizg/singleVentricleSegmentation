import sys
import os
import argparse
# import configparser
import torch    

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import torch_utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu', type=str, default="0")
    args = parser.parse_args()

    DEVICE = torch_utils.getTorchDevice(args.gpu)
    
    test = torch.zeros([7, 8, 9, 3]).float().to(DEVICE)

    # if torch.cuda.is_available():

    #     DEVICE = torch.device('cuda:' + str(args.gpu))

    #     # test
    #     device_count = torch.cuda.device_count()
    #     device_name = torch.cuda.get_device_name(range(device_count))
    #     current_device = torch.cuda.current_device()
    #     print("=================================")
    #     print("     TEST GPU:  ")
    #     print("device_count = ", device_count)
    #     print("device_name = ", device_name)
    #     print("current_device = ", current_device)
    #     print("=================================")
    # else:
    #     DEVICE = 'cpu'
    #     #torch.cuda.set_device(DEVICE)
    
    # print("=================================")
    # print("     TEST CPU:  ")
    # print("totally available CPUs = ", torch.get_num_threads() )
    # torch.set_num_threads(8)
    # print("use num CPUs = ", torch.get_num_threads() )
    # print("=================================")