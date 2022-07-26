import shutil
import os
import os.path as osp

if __name__ == "__main__":
    copy_ntimes = 1
    flow = 'forward'
    # flow = 'backward'
    src_path = osp.join('data/leftVentricleData_split/optical_flow', flow)
    dst_path = osp.join('data/leftVentricleData_split_SC_x2/optical_flow', flow)

    src_dirs = os.listdir(src_path)

    for dir in src_dirs:
        src_patient_path = osp.join(src_path, dir)
        if osp.isdir(src_patient_path):
            dst_patient_path = osp.join(dst_path, dir)
            shutil.copytree(src_patient_path, dst_patient_path)

            for n in range(copy_ntimes):
                dst_patient_path = osp.join(dst_path, dir + f'_A_{n}')
                shutil.copytree(src_patient_path, dst_patient_path)
