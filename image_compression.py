#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 19:52:25 2023

@author: hboateng
"""
from skimage import color
from numpy import linalg as la 
from numpy import dstack
from numpy import random as rdm


class imgComp:
    
    def __init__(self,img,rank,p,q):
        self.img = img
        self.rank   = rank
        self.p = p  # oversampling parameter
        self.q = q  # power iterations
        
    def rsvd(self,A):   ## randomized SVD
    # step 1: Sample column space of A with P matrix
        m = A.shape[1]  # number of rows of matrix A
        P = rdm.randn(m,self.rank+self.p) # Gaussian random matrix
        Z = A@P  # Sketch (sampling column space of A )
        
        # Power iterations
        for k in range(1,self.q+1):
            Z = A@(A.T@Z)
            
        #QR factorization to get Q
        Q,R = la.qr(Z)
        
    # step 2: Compute svd on projection:  B = Q'*A
        B = Q.T@A
        ub,s,vt = la.svd(B,full_matrices=False)
        return Q@ub, s, vt
        
        
    def graycomp(self):
        # convert color image to grayscale
        if self.img.shape[2] == 3:
            A = color.rgb2gray(self.img) #grayscale
        else:
            A = self.img
            
        U,S,Vt = la.svd(A,full_matrices=False) #svd of grayscale
        
        s = self.rank
        #print(f"rank of image = {la.matrix_rank(A)}")
        return A, (U[:,0:s]*S[0:s])@Vt[0:s,:]
    
    def rgraycomp(self): # compression using randomized svd
        # convert color image to grayscale
        if self.img.shape[2] == 3:
            A = color.rgb2gray(self.img) #grayscale
        else:
            A = self.img
            
        U,S,Vt = self.rsvd(A) #randomized svd of grayscale
        
        s = self.rank
        
        return (U[:,0:s]*S[0:s])@Vt[0:s,:]
    
    def colorcomp(self):
        if self.img.shape[2] == 3:
            R = self.img[:, :, 0]  #red
            G = self.img[:, :, 1]  #green
            B = self.img[:, :, 2]  #blue
            
            Ur,Sr,Vrt = la.svd(R,full_matrices=False)    #svd of red comp
            Ug,Sg,Vgt = la.svd(G,full_matrices=False)  #svd of green comp
            Ub,Sb,Vbt = la.svd(B,full_matrices=False)  #svd of blue comp
            
            s     = self.rank
            rcomp = Ur[:,0:s]*Sr[0:s]@Vrt[0:s,:]
            gcomp = Ug[:,0:s]*Sg[0:s]@Vgt[0:s,:]
            bcomp = Ub[:,0:s]*Sb[0:s]@Vbt[0:s,:]
            
            compimg = dstack((rcomp,gcomp,bcomp))
        else:
            print("Image is not a colored image")
            compimg = self.img
        
            
        return self.img, compimg
    
    def rcolorcomp(self):
        if self.img.shape[2] == 3:
            R = self.img[:, :, 0]  #red
            G = self.img[:, :, 1]  #green
            B = self.img[:, :, 2]  #blue
            
            Ur,Sr,Vrt = self.rsvd(R)    #randomized svd of red comp
            Ug,Sg,Vgt = self.rsvd(G)    #randomized svd of green comp
            Ub,Sb,Vbt = self.rsvd(B)    #randomized svd of blue comp
            
            s     = self.rank
            rcomp = Ur[:,0:s]*Sr[0:s]@Vrt[0:s,:]
            gcomp = Ug[:,0:s]*Sg[0:s]@Vgt[0:s,:]
            bcomp = Ub[:,0:s]*Sb[0:s]@Vbt[0:s,:]
            
            compimg = dstack((rcomp,gcomp,bcomp))
        else:
            print("Image is not a colored image")
            compimg = self.img
        
            
        return compimg

