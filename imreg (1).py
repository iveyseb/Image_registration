import imreg_master
import numpy as np
import matplotlib.pyplot as plt
import glob
from imreg_master import imreg
import cv2
import time
import matplotlib
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from skimage.measure import compare_ssim
from sklearn.metrics import mutual_info_score
from texttable import Texttable
from skimage import io, img_as_float
import statistics
from matplotlib import cm
from PIL import Image
from texttable import Texttable
import pandas as pd
import xlsxwriter
from PIL import Image
import argparse
import imreg_dft as ird
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import AsinhStretch
# file:///media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/input_files/AIH_scene01_00R_DAPI_DAPI.tif
#directory = r'/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_qc_images'
# os.chdir(directory)
parser = argparse.ArgumentParser(description='Registration')
parser.add_argument("--fixed",
                    help="path to input fixed image.",
                    type=str)
parser.add_argument("--moving",
                    help="path to input moving image.",
                    type=str)
parser.add_argument("--stats",
                    help="path to output stats file.",
                    type=str)
parser.add_argument("--out_tm",
                    help="path to output transformation matrix.",
                    type=str)
parser.add_argument("--out_qc",
                    help="path to output quality control plots folder.",
                    type=str)
parser.add_argument("--out_ti",
                    help="path to output file.",
                    type=str)
args = parser.parse_args()
# print(len(args))
print(args)


def get_regstat(fixed_image, moving_image, out_stats, out_tm, out_qc, tm):
    x = [[]]

    time1 = time.time()
    # loading images
    target = plt.imread(fixed_image)
    #tar=cv2.imread(fixed_image)
    target_name = os.path.split(fixed_image)[1]
    # print(target_name)
    # target=cv2.resize(target,(5705,5705))
    # tar=cv2.resize(tar,(5705,5705))
    moving = plt.imread(moving_image)
    #mov=cv2.imread(moving_image)
    # moving=cv2.resize(moving,(5705,5705))
    # mov=cv2.resize(mov,(5705,5705))
    moving_name = os.path.split(moving_image)[-1]
    # print(moving_name)

    # registration
    #a=target.shape[0]
    b=target.shape[1]
    #moving=cv2.resize(moving,(b,b))
    mov_avg=np.mean(moving.astype("float"))
    print(target.shape)
    print(moving.shape)
    res=ird.similarity(target,moving)
    def_moving=res['timg']
    scale=res['scale']
    angle=res['angle']
    (t0,t1)=res['tvec']
    #def_moving, scale, angle, (t0, t1) = imreg.similarity(target, moving)
    def_mov_array=imreg.similarity_matrix(scale,angle,(t0,t1))
    np.save(out_tm, def_mov_array)
    def_avg=np.mean(def_moving.astype("float"))
    # def_moving=cv2.resize(def_moving,(5705,5705))

    time2 = time.time()
    ti = time2-time1
    # statistical results
    cc = np.corrcoef(moving.flat, target.flat)
    r1 = cc[0, 1]

    score1, diff = compare_ssim(target, moving, full=True, multichannel=True)

    loss_before = 0.5*(1-r1) + 0.5*(1-score1)

    cc = np.corrcoef(def_moving.flat, target.flat)
    r2 = cc[0, 1]

    score2, diff = compare_ssim(def_moving, target, full=True)

    loss_after = 0.5*(1-r2) + 0.5*(1-score2)

    mi_bef = mutual_info_score(moving.ravel(), target.ravel())

    mi_aft = mutual_info_score(target.ravel(), def_moving.ravel())

    mae = np.sum(np.absolute(target.astype("float") - moving.astype("float")))
    u1 = np.sum(target.astype("float") + moving.astype("float"))
    mean_before = np.divide(mae, u1)
    u2 = np.sum(target.astype("float") + def_moving.astype("float"))
    mae = np.sum(np.absolute(target.astype(
        "float") - def_moving.astype("float")))
    mean_after = np.divide(mae, u2)
    t = Texttable()
    #print(def_moving.shape)
    if mi_aft > mi_bef:
        plt.imsave(tm, def_moving,cmap='gray')
    else:
        plt.imsave(tm, moving,cmap='gray')
    # plt.imsave(tm,def_moving,cmap='gray')
    x.append([target_name, moving_name, r1, r2, score1, score2, loss_before,
              loss_after, mi_bef, mi_aft, mean_before, mean_after,mov_avg,def_avg, ti])

    t.add_rows(x)
    t.set_cols_align(['r', 'r', 'r', 'r', 'r', 'r',
                      'r', 'r', 'r', 'r', 'r', 'r', 'r','r', 'r'])
    t.header(['TARGET', 'MOVING', 'PCC_BEFORE', 'PCC_AFTER', 'SSIM_BEFORE', 'SSIM_AFTER',
              'loss_BEFORE', 'loss_AFTER', 'MI_BEFORE', 'MI_AFTER', 'MEAN_BEFORE', 'MEAN_AFTER', 'AVERAGE_INTENSITY_BEFORE', 'AVERAGE_INTENSITY_AFTER','TIME'])
    def_mov1 = cv2.imread(tm,0)

    # plotting the results

    transform = AsinhStretch() + AsymmetricPercentileInterval(0.1, 99.99)
    targ= transform(target)
    #targ=cv2.cvtColor(targ.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    #targ_im=Image.fromarray(targ)

    plt.imsave('target.tiff', targ, cmap='gray')
    #plt.imsave(r'/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_images/target.tiff', targ, cmap='gray')
    movi = transform(moving)
    #movi=cv2.cvtColor(movi.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    plt.imsave('moving.tiff', movi, cmap='gray')
    #movi_im=Image.fromarray(movi)
    #plt.imsave(r'/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_images/moving.tiff', movi, cmap='gray')
    targ = cv2.imread('target.tiff')
    #targ=Image.open(targ_im)
    movi= cv2.imread('moving.tiff')
    #movi=Image.open(movi_im)
    #targ = cv2.imread( r'/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_images/target.tiff')
    #movi = cv2.imread(r'/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_images/moving.tiff')
    
    # def_mov1=cv2.resize(def_mov1,(5705,5705))
    def_mov1 = transform(def_mov1)
    #print(def_mov2.shape)
    
    #def_mov1=cv2.cvtColor(def_mov1.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    plt.imsave('def_moving.tiff', def_mov1, cmap='gray')
    #def_mov1=Image.fromarray(def_mov1)
    #plt.imsave('/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_images/def_moving.tiff', def_mov1, cmap='gray')
    def_mov1 = cv2.imread('def_moving.tiff')
    #def_mov1=Image.open(def_mov1)
   
    #def_mov1 = cv2.imread(r'/media/u0132399/L-drive/GBW-0075_MILAN_Multiplex/u0140317/registration_comparison/imreg/output_images/def_moving.tiff')
    #targ = targ.astype(np.uint8)
    #movi = movi.astype(np.uint8)
    #def_mov1 = def_mov1.astype(np.uint8)
    
    tar1 = np.zeros(targ.shape)
    mov1 = np.zeros(movi.shape)
    def_mov = np.zeros(def_mov1.shape)
    tar1[:, :, 0] = targ[:, :, 0]
    mov1[:, :, 2] = movi[:, :, 2]
    def_mov[:, :, 2] = def_mov1[:, :, 2]
    tar1 = tar1.astype(np.uint8)
    mov1 = mov1.astype(np.uint8)
    def_mov = def_mov.astype(np.uint8)
  
    # mov[np.where((mov>[15]).all(axis=2))]=[255,0,0]
    # def_mov1[np.where((def_mov1>[15]).all(axis=2))]=[255,0,0]
    alpha = 1
    beta = 1
    gamma = 0
    t1 = cv2.addWeighted(tar1, alpha, mov1, beta, gamma)
    
    # filename=target_name+moving_name+'unreg_imreg.tif'
    # cv2.imwrite(,t1)
    
    
    t2 = cv2.addWeighted(tar1, alpha, def_mov, beta, gamma)
    
    # filename=target_name+moving_name+'reg_imreg.tif'
    # cv2.imwrite(filename,t2)
    if mi_aft > mi_bef:
        cv2.imwrite(out_qc, t2)
        
    else:
        cv2.imwrite(out_qc, t1)
        
    # plt.figure(figsize=(10,10))
    # plt.axis('off')
    # plt.imshow(t1,cmap='gray')
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()
    # plt.figure(figsize=(10,10))
    # plt.axis('off')
    # plt.imshow(t2,cmap='gray')
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()

    df = pd.DataFrame(x, columns=['TARGET', 'MOVING', 'PCC_BEFORE', 'PCC_AFTER', 'SSIM_BEFORE', 'SSIM_AFTER',
                                  'loss_BEFORE', 'loss_AFTER', 'MI_BEFORE', 'MI_AFTER', 'MEAN_BEFORE', 'MEAN_AFTER','AVERAGE_INTENSITY_BEFORE', 'AVERAGE_INTENSITY_AFTER', 'TIME'])
    #df[columns]= ['PCC_BEFORE','PCC_AFTER','SSIM_BEFORE' ,'SSIM_AFTER','loss_BEFORE','loss_AFTER','MI_BEFORE','MI_AFTER','MEAN_BEFORE','MEAN_AFTER']
    #writer = pd.ExcelWriter(out_stats)
    df.to_csv(out_stats, index=None, header=True)
    # writer.save()
    #print(t.draw())

    return t


#fixed_image = '/media/u0132399/LaCie_MVH/MILAN2020/Data/04_preprocessed/AIH/scene01/AIH_scene01_00R_DAPI_DAPI.tif'
#moving_image = '/media/u0132399/LaCie_MVH/MILAN2020/Data/04_preprocessed/AIH/scene01/AIH_scene01_21R_DAPI_DAPI.tif'
#out_stats = '/media/u0132399/LaCie_MVH/MILAN2020/Results/05_registration_stats/AIH/scene01/AIH_scene01_21R.csv'
#out_tm = '/media/u0132399/LaCie_MVH/MILAN2020/Results/05_registration_transformation_matrices/AIH/scene01/AIH_scene01_21R.npy'
#out_qc = '/media/u0132399/LaCie_MVH/MILAN2020/Results/05_registration_plots_all/AIH/scene01/AIH_scene01_21R.tiff'
#tm = '/media/u0132399/LaCie_MVH/MILAN2020/Data/05_registered_images/AIH/scene01/AIH_scene01_21R_DAPI_DAPI.tiff'

#t = get_regstat(fixed_image=fixed_image,  # args.fixed, # input fixed image
#                moving_image=moving_image,  # args.moving, # input moving image
#                out_stats=out_stats,  # args.stats, # output stats file
#               out_tm=out_tm,  # args.out_tm, # output transformation matrix
#                out_qc=out_qc,  # args.out_qc, # output folder qc plots
#                tm=tm  # args.out_ti # output file
#                )

t=get_regstat(fixed_image= args.fixed, # input fixed image
 moving_image = args.moving, # input moving image
 out_stats = args.stats, # output stats file
 out_tm = args.out_tm, # output transformation matrix
 out_qc = args.out_qc, # output folder qc plots
 tm= args.out_ti # output file
 )
