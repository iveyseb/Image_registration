import imreg_master
import numpy as np
import matplotlib.pyplot as plt
import glob
from imreg_master import imreg
import cv2
import time
import matplotlib
import os
import argparse
import imreg_dft as ird


parser = argparse.ArgumentParser(description='Transformation')
parser.add_argument("--moving",
                    help="path to input moving image.",
                    type=str)
parser.add_argument("--transformation_matrix",
                    help="path to transformation_matrix.",
                    type=str)

parser.add_argument("--transformed_image",
                    help="path to transformed_image.",
                    type=str)

args = parser.parse_args()
#print(args)


def transformation(moving,trans_mat,trans_im):
    print(moving)
    print(trans_mat)
    print(trans_im)
    mov=plt.imread(moving)
    print(mov.shape)
    trans=np.load(trans_mat)
    scale=trans[0]
    angle=trans[1]
    t1=trans[2]
    t2=trans[3]
    tvec=np.array([t1,t2])
    trans_image=ird.transform_img(mov,scale,angle,tvec)
    
    plt.imsave(trans_im,trans_image,cmap='gray')
    



    return(trans_image)

#fixed='/media/u0132399/LaCie_MVH/MILAN2020/Data/04_preprocessed/ASH/scene110/ASH_scene110_00R_TRITC_BLANK.tif'
#moving='/media/u0132399/LaCie_MVH/MILAN2020/Data/04_preprocessed/ASH/scene110/ASH_scene110_29R_TRITC_BLANK.tif'
#transformation_matrix='/media/u0132399/LaCie_MVH/MILAN2020/Results/05_registration_transformation_matrices/ASH/scene110/ASH_scene110_29R.npy'
#transformed_image='/media/u0132399/LaCie_MVH/MILAN2020/Data/05_registered_images/ASH/scene110/ASH_scene110_29R_TRITC_BLANK.tiff'
t=transformation(moving=args.moving,trans_mat=args.transformation_matrix,  
              trans_im=args.transformed_image)
#t=transformation(fixed=fixed,moving=moving,trans_mat=transformation_matrix,trans_im=transformed_image)

