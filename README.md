# Image_registration
Registration of 2d medical images(microscopy) using FFT (module named Imreg). Convert Images to fourier domain where correaltion between them is calculated from which different shifts are stimated. Implemenatataion  using https://github.com/matejak/imreg_dft

Given two images, imreg calculates difference between scale, rotation and position of imaged features. The code is a function that can be used by orchestrator for parellel processing! 
Inputs: input_fixed_image & input_moving_image
Outputs: output_stats_file, outut_transformation_matrix, output_qc_plots & output_transformed_image

This function called by orchestrator code where the name of this script is to be mentioned along with the csv file where the locations of all the function parameters should also be given as input.

