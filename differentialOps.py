import numpy as np
import math
import torch
# from colorama import init
from termcolor import colored


def printColoredError( diff, tol=1.e-5, accTol=1.e-2 ):
    if( diff < tol):
        print(colored(diff, 'green'))
    elif( diff < accTol ):
        print(colored(diff, 'yellow'))
    else:
        print(colored(diff, 'red'))


# forward difference quotients in 1d
class Nabla1D_Forward:        
    
    def forward(self, u):
        device = u.device
        NX = u.size(dim=0)
        p = torch.zeros((NX,1), device=device)
        p[:-1,0] += (u[1:]-u[:-1])
        return p        
        
    def backward(self, p):
        device = p.device
        NX, _ = p.shape
        u = torch.zeros((NX), device=device)
        u[1:]  += p[:-1,0]
        u[:-1] -= p[:-1,0]
        return u
    
    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla1D_Forward:", end=" " )
        u = torch.randn(*size_in)
        p = torch.randn(*size_out)
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs-rhs)).item()
        printColoredError(diff)


# forward difference quotients in 2d
class Nabla2D_Forward:        
    
    def forward(self, u):
        device = u.device
        NY = u.size(dim=0)
        NX = u.size(dim=1)
        p = torch.zeros((NY,NX,2), device=device)
        p[:,:-1,0] += (u[:,1:]-u[:,:-1])
        p[:-1,:,1] += (u[1:,:]-u[:-1,:])
        return p        
        
    def backward(self, p):
        device = p.device
        NY,NX, _ = p.shape
        u = torch.zeros((NY,NX), device=device)

        u[:,1:]  += p[:,:-1,0]
        u[:,:-1] -= p[:,:-1,0]

        u[1:,:]  += p[:-1,:,1]
        u[:-1,:] -= p[:-1,:,1]

        return u
    
    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla2D_Forward", end=" ")
        u = torch.randn(*size_in)
        p = torch.randn(*size_out)
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs-rhs)).item()
        printColoredError(diff)


# forward difference quotients in 3D
class Nabla3D_Forward:        
    def forward(self, u):
        device = u.device
        NZ = u.size(dim=0)
        NY = u.size(dim=1)
        NX = u.size(dim=2)
        hZ = NZ / math.sqrt(0.5*NX*NX+0.5*NY*NY)
        p = torch.zeros((NZ,NY,NX,3), device=device)
        p[:,:,:-1,0] += u[:,:,1:]-u[:,:,:-1]
        p[:,:-1,:,1] += u[:,1:,:]-u[:,:-1,:]
        p[:-1,:,:,2] += (u[1:,:,:]-u[:-1,:,:])/hZ
        return p        
        
    def backward(self, p):
        device = p.device
        NZ,NY,NX, _ = p.shape
        hZ = NZ / math.sqrt(0.5*NX*NX+0.5*NY*NY)
        u = torch.zeros((NZ, NY, NX), device=device)

        u[:,:,1:]  += p[:,:,:-1,0]
        u[:,:,:-1] -= p[:,:,:-1,0]

        u[:,1:,:] += p[:,:-1,:,1]
        u[:,:-1,:] -= p[:,:-1,:,1]

        u[1:,:,:] += p[:-1,:,:,2]/hZ
        u[:-1,:,:] -= p[:-1,:,:,2]/hZ

        return u
    
    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla3D_Forward:", end=" ")
        u = torch.randn(*size_in)
        p = torch.randn(*size_out)
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs-rhs)).item()
        printColoredError(diff) 



# central difference quotients in 1D
class Nabla1D_Central:        

    def forward(self, u):
        device = u.device
        NX = u.size(dim=0)
        p = torch.zeros((NX,1), device=device)
        p[0,0]=0.5*(u[1]-u[0])
        p[1:-1,0] = 0.5*(u[2:]-u[:-2])
        p[-1,0]=0.5*(u[-1]-u[-2])
        return p      
        
    def backward(self, p):
        device = p.device
        NX, _ = p.shape
        u = torch.zeros((NX), device=device)
        u[0] -= 0.5*p[0,0] 
        u[1:]  += 0.5*p[:-1,0]
        u[:-1] -= 0.5*p[1:,0]
        u[-1] += 0.5*p[-1,0]
        return u
    
    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla1D_Central:", end=" ")
        u = torch.randn(*size_in)
        p = torch.randn(*size_out)
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs-rhs)).item()
        printColoredError(diff)
            

# central difference quotients in 2D
class Nabla2D_Central:        

    def forward(self, u):
        device = u.device
        NY = u.size(dim=0)
        NX = u.size(dim=1)
        p = torch.zeros((NY,NX,2), device=device)

        p[:,0,0]=0.5*(u[:,1]-u[:,0])
        p[:,1:-1,0] = 0.5*(u[:,2:]-u[:,:-2])
        p[:,-1,0]=0.5*(u[:,-1]-u[:,-2])

        p[0,:,1]=0.5*(u[1,:]-u[0,:])
        p[1:-1,:,1] = 0.5*(u[2:,:]-u[:-2,:])
        p[-1,:,1]=0.5*(u[-1,:]-u[-2,:])

        return p      
        
    def backward(self, p):
        device = p.device
        NY,NX, _ = p.shape
        u = torch.zeros((NY,NX), device=device)

        u[:,0] -= 0.5*p[:,0,0] 
        u[:,1:]  += 0.5*p[:,:-1,0]
        u[:,:-1] -= 0.5*p[:,1:,0]
        u[:,-1] += 0.5*p[:,-1,0]

        u[0,:] -= 0.5*p[0,:,1] 
        u[1:,:]  += 0.5*p[:-1,:,1]
        u[:-1,:] -= 0.5*p[1:,:,1]
        u[-1,:] += 0.5*p[-1,:,1]

        return u
    
    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla2D_Central:", end=" ")
        u = torch.randn(*size_in)
        p = torch.randn(*size_out)
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs-rhs)).item()
        printColoredError(diff)


# central difference quotients in 3D
class Nabla3D_Central:        
    def forward(self, u):
        device = u.device
        NZ = u.size(dim=0)
        NY = u.size(dim=1)
        NX = u.size(dim=2)
        hZ = NZ / math.sqrt(0.5*NX*NX+0.5*NY*NY)
        p = torch.zeros((NZ,NY,NX,3), device=device)

        p[:,:,0,0] += 0.5*(u[:,:,1]-u[:,:,0])
        p[:,:,1:-1,0] += 0.5*(u[:,:,2:]-u[:,:,:-2])
        p[:,:,-1,0] += 0.5*(u[:,:,-1]-u[:,:,-2])

        p[:,0,:,1] += 0.5*(u[:,1,:]-u[:,0,:])
        p[:,1:-1,:,1] += 0.5*(u[:,2:,:]-u[:,:-2,:])
        p[:,-1,:,1] += 0.5*(u[:,-1,:]-u[:,-2,:])

        p[0,:,:,2] += 0.5*(u[1,:,:]-u[0,:,:])/hZ
        p[1:-1,:,:,2] += 0.5*(u[2:,:,:]-u[:-2,:,:])/hZ
        p[-1,:,:,2] += 0.5*(u[-1,:,:]-u[-2,:,:])/hZ

        return p        
        
    def backward(self, p):
        device = p.device
        NZ, NY, NX, _ = p.shape
        hZ = NZ / math.sqrt(0.5*NX*NX+0.5*NY*NY)
        u = torch.zeros((NZ, NY, NX), device=device)

        u[:,:,0] -= 0.5*p[:,:,0,0] 
        u[:,:,1:]  += 0.5*p[:,:,:-1,0]
        u[:,:,:-1] -= 0.5*p[:,:,1:,0]
        u[:,:,-1] += 0.5*p[:,:,-1,0]

        u[:,0,:] -= 0.5*p[:,0,:,1] 
        u[:,1:,:]  += 0.5*p[:,:-1,:,1]
        u[:,:-1,:] -= 0.5*p[:,1:,:,1]
        u[:,-1,:] += 0.5*p[:,-1,:,1]

        u[0,:,:] -= 0.5*p[0,:,:,2]/hZ
        u[1:,:,:]  += 0.5*p[:-1,:,:,2]/hZ
        u[:-1,:,:] -= 0.5*p[1:,:,:,2]/hZ
        u[-1,:,:] += 0.5*p[-1,:,:,2]/hZ

        return u
    
    def check_adjointness(self, size_in, size_out):
        print("check adjointness of Nabla3D_Central:", end=" ")
        u = torch.randn(*size_in)
        p = torch.randn(*size_out)
        lhs = self.forward(u).reshape(-1).dot(p.reshape(-1))
        rhs = self.backward(p).reshape(-1).dot(u.reshape(-1))
        diff = torch.max(torch.abs(lhs-rhs)).item()
        printColoredError(diff)


