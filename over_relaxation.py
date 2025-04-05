# Licenced with Monzilla Public Licence 2.0
"""
Gaussian Over-Relaxiation Method
"""

import numpy as np
import matplotlib.pyplot as plt


class OverRelaxation:
    
    def __init__(self,size, spacing):
        
        self.N = size
        self.h = spacing


