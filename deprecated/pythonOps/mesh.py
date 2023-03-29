import numpy as np
import math
import torch


class MeshInfo1D:
    def __init__(self, NX, LX): 
        self.NX = NX
        self.LX = LX
        self.hX = LX/(NX-1)

class MeshInfo2D:
    def __init__(self, NY, NX, LY, LX): 
        self.NY = NY
        self.NX = NX
        self.LY = LY
        self.LX = LX
        self.hY = LY/(NY-1)
        self.hX = LX/(NX-1)


class MeshInfo3D:
    def __init__(self, NZ, NY, NX, LZ, LY, LX): 
        self.NZ = NZ
        self.NY = NY
        self.NX = NX
        self.NZ = NZ
        self.LY = LY
        self.LX = LX
        self.hZ = LZ/(NZ-1)
        self.hY = LY/(NY-1)
        self.hX = LX/(NX-1)

    