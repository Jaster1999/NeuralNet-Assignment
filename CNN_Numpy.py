import numpy as np
import cv2
import sys

def conv(img, conv_filter):  
    filter_size = conv_filter.shape[0]  
    result = np.zeros((img.shape))  
    #Looping through the image to apply the convolution operation.  
    for r in np.uint16(np.arange(filter_size/2, img.shape[0]-filter_size/2-2)):  
        for c in np.uint16(np.arange(filter_size/2, img.shape[1]-filter_size/2-2)):  
            #Getting the current region to get multiplied with the filter.  
            curr_region = img[r:r+filter_size, c:c+filter_size]  
            #Element-wise multipliplication between the current region and the filter.  
            curr_result = curr_region * conv_filter  
            conv_sum = np.sum(curr_result) #Summing the result of multiplication.  
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.  
               
    # Clipping the outliers of the result matrix.  
    final_result = result[np.uint16(filter_size/2):result.shape[0]-np.uint16(filter_size/2), np.uint16(filter_size/2):result.shape[1]-np.uint16(filter_size/2)]  
    return final_result  

def convolution(img, conv_filter):  
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.  
        if img.shape[-1] != conv_filter.shape[-1]:  
            print("Error: Number of channels in both image and filter must match.")  
            sys.exit()  
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.  
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')  
        sys.exit()  
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.  
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')  
        sys.exit()  
  
    # An empty feature map to hold the output of convolving the filter(s) with the image.  
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1,   
                                img.shape[1]-conv_filter.shape[1]+1,   
                                conv_filter.shape[0]))  
  
    # Convolving the image by the filter(s).  
    for filter_num in range(conv_filter.shape[0]):  
        print("Filter ", filter_num + 1)  
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.  
        """  
        Checking if there are mutliple channels for the single filter. 
        If so, then each channel will convolve the image. 
        The result of all convolutions are summed to return a single feature map. 
        """  
        if len(curr_filter.shape) > 2:  
            conv_map = conv(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.  
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.  
                conv_map = conv_map + conv(img[:, :, ch_num], curr_filter[:, :, ch_num])  
        else: # There is just a single channel in the filter.  
            conv_map = conv(img, curr_filter)  
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps. 

def main():


    l1_filter = np.zeros((2,3,3))
    l1_feature_map = conv(img, l1_filter)  


if __name__ == '__main__':
    main()

# https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
# https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1 