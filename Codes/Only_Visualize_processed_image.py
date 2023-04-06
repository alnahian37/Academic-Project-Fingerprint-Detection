# %%
import numpy as np
import os
import pickle
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import matplotlib.image as mpimg
import skimage
from skimage.filters import threshold_otsu as otsu
from skimage.filters import unsharp_mask as unsharp
from skimage.metrics import structural_similarity as ssim
import re

# %%


parent_dir=os.getcwd()

test_dir = os.path.join(parent_dir, 'Project Dataset/Test/')



def getTerminationBifurcation(img, mask):
    img = img == 255;
    (rows, cols) = img.shape;
    minutiaeTerm = np.zeros(img.shape);
    minutiaeBif = np.zeros(img.shape);
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if(img[i][j] == 1):
                block = img[i-1:i+2,j-1:j+2];
                block_val = np.sum(block);
                if(block_val == 2):
                    minutiaeTerm[i,j] = 1;
                elif(block_val == 4):
                    minutiaeBif[i,j] = 1;
    
    mask = convex_hull_image(mask>0)
 
    mask = erosion(mask, square(2))     
    
    minutiaeTerm = np.uint8(mask)*minutiaeTerm
    minutiaeBif = np.uint8(mask)*minutiaeBif
    return(minutiaeTerm, minutiaeBif)

# %%


def ShowResults(skel, TermLabel, BifLabel,st=''):
    minutiaeBif = TermLabel * 0;
    
    minutiaeTerm = BifLabel * 0;

    (rows, cols) = skel.shape;
    DispImg = np.zeros((rows, cols, 3), np.uint8);
   
    DispImg[:, :, 1] = skel;
    
    
    RP = skimage.measure.regionprops(BifLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeBif[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 2)
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
    

    RP = skimage.measure.regionprops(TermLabel)
    for idx, i in enumerate(RP):
        (row, col) = np.int16(np.round(i['Centroid']))
        minutiaeTerm[row, col] = 1
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 2)
        #skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255));
    
    plt.figure(figsize=(10,10))
    plt.title("Minutiae extraction results for "+st)
    plt.imshow(DispImg)
    #plt.show()

    return minutiaeBif, minutiaeTerm


# %%
def process_image(img_name,im_name):
    img = cv2.imread(img_name)
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(256,256))
    img=img[16:img.shape[0]-16,16:img.shape[1]-16]
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title("Original Image for "+im_name)

    img=unsharp(img,radius=3,amount=2)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title("Unsharp masking for "+im_name)
    THRESHOLD1=otsu(img)
   

    img = np.array(img > THRESHOLD1).astype(int)
    #invert 1 to 0 and 0 to 1
    img = 1 - img
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title("Thresholding for "+im_name)


    
    skel = skimage.morphology.skeletonize(img)
    skel =skel*255
    plt.figure(figsize=(10,10))
    plt.imshow(skel)
    plt.title("Skeletonized Image for "+im_name)

    #plt.show()

    
    mask = img*255


    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)
    return minutiaeTerm, minutiaeBif,skel

# %%

finger_files=os.listdir(test_dir)
#finger_files.sort()


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

finger_files=sorted_alphanumeric(finger_files)
print(finger_files[0])
while 1:
    msg="\n\nEnter a number between 1 and "+str(len(finger_files))+': press ctrl+c to quit\n'
    val=input(msg)
    
    img_name=finger_files[int(val)-1]
    print(img_name)

    img_full_name=test_dir+img_name
    #print(img_full_name)
    mt,mb,sk=process_image(img_full_name,img_name)

    BifLabel = skimage.measure.label(mb, connectivity=1);
    TermLabel = skimage.measure.label(mt, connectivity=1);

    ShowResults(sk, TermLabel, BifLabel,st=img_name)
    plt.figure(figsize=(10,10))

    plt.imshow(mb)
    plt.title("Minutiae Locations (Bifurcation) for "+img_name)
    plt.show()
   



