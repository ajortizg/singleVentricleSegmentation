import nibabel as nib
import numpy as np
import torch
import os

from utilities.config import *
from TVL1OF.tvl1Scipy import TVL1Scipy
from TVL1OF.tvl1Cuda import *

pythonOps_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pythonOps'))
sys.path.append(pythonOps_lib_path)
import mesh

if __name__ == "__main__":
    # np.set_printoptions(precision=2, suppress=True)

    # # Load 4D nifty [x,y,z,t]
    # vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_FILE]))
    # data = vol.get_fdata()
    # print(f"Dims: {data.ndim}, shape: {data.shape}, type: {data.dtype}")

    # #==================================
    # # Load 4D nifty [x,y,z,t]
    # vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_FILE]))
    # nii_data_xyzt = vol.get_fdata()
    # print(f"Dims: {nii_data_xyzt.ndim}, shape: {nii_data_xyzt.shape}, type: {nii_data_xyzt.dtype}")
    # NX = nii_data_xyzt.shape[0]
    # NY = nii_data_xyzt.shape[1]
    # NZ = nii_data_xyzt.shape[2]
    # NT = nii_data_xyzt.shape[3]

    # #==================================
    # #scaling of data 
    # totalMinValue = np.amin(nii_data_xyzt)
    # totalMaxValue = np.amax(nii_data_xyzt)
    # print(f"input (min,max) = {totalMinValue,totalMaxValue}")
    # # scaleMaxValue = 255.
    # scaleMaxValue = 1.
    # print("scaling of data to max value", scaleMaxValue)
    # nii_data_xyzt *= scaleMaxValue / totalMaxValue

    # ## swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    # print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
    # nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    # print( f"dimension after swap: (Z,Y,X,T) = {nii_data.shape}")

    # # Get two adjacent volumes and change axes to z,y,x
    # # t0, t1 = 8, 9
    # # # I0 = nii_data[:12, :, :, t0]
    # # # I1 = nii_data[:12, :, :, t1]
    # # I0 = nii_data[:, :, :, t0]
    # # I1 = nii_data[:, :, :, t1]


    #
    NZ = 14
    NY = 27
    NX = 34
    LZ = NZ-1
    LY = NY-1
    LX = NX-1
    meshInfo3D_python = mesh.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)
    meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)

    # 
    algScipy = TVL1Scipy()
    algCuda = TVL1OpticalFlowCuda()

    u = torch.randn([NZ,NY,NX,3]).float().to(DEVICE)
    v = torch.randn([NZ,NY,NX,3]).float().to(DEVICE)
    p = torch.randn([NZ,NY,NX,3,3]).float().to(DEVICE)

    uold = u.detach().clone()
    vold = v.detach().clone()
    pold = p.detach().clone()

    ##
    print("\nupdate dual variable with cuda:")
    algCuda.updateDualVariable(u,v,p,meshInfo3D_cuda)
    print("diff u = ", torch.norm(u - uold).item() )
    print("diff v = ", torch.norm(v - vold).item() )
    print("diff p = ", torch.norm(p - pold).item() )

    ##
    u_torch = uold.detach().clone()
    v_torch = vold.detach().clone()
    p_torch = pold.detach().clone()
    u_np = u_torch.cpu().detach().numpy()
    v_np = v_torch.cpu().detach().numpy()
    p1_np = p_torch[:,:,:,:,0].cpu().detach().numpy()
    p2_np = p_torch[:,:,:,:,1].cpu().detach().numpy()
    p3_np = p_torch[:,:,:,:,2].cpu().detach().numpy()
    print("\nupdate dual variable with python:")
    u_np = algScipy.updateDualVariable(u_np,v_np,p1_np,p2_np,p3_np,meshInfo3D_python)
    print("diff u = ", torch.norm( torch.from_numpy(u_np).float().to(DEVICE) - uold).item() )
    print("diff v = ", torch.norm( torch.from_numpy(v_np).float().to(DEVICE) - vold).item() )
    print("diff p1 = ", torch.norm( torch.from_numpy(p1_np).float().to(DEVICE) - pold[:,:,:,:,0]).item() )
    print("diff p2 = ", torch.norm( torch.from_numpy(p2_np).float().to(DEVICE) - pold[:,:,:,:,1]).item() )
    print("diff p3 = ", torch.norm( torch.from_numpy(p3_np).float().to(DEVICE) - pold[:,:,:,:,2]).item() )

    ##
    print("\ndiff python vs cuda:")
    print("diff u = ", torch.norm( torch.from_numpy(u_np).float().to(DEVICE) - u).item() )
    print("diff v = ", torch.norm( torch.from_numpy(v_np).float().to(DEVICE) - v).item() )
    print("diff p1 = ", torch.norm( torch.from_numpy(p1_np).float().to(DEVICE) - p[:,:,:,:,0]).item() )
    print("diff p2 = ", torch.norm( torch.from_numpy(p2_np).float().to(DEVICE) - p[:,:,:,:,1]).item() )
    print("diff p3 = ", torch.norm( torch.from_numpy(p3_np).float().to(DEVICE) - p[:,:,:,:,2]).item() )