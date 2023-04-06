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




from skimage.util import invert

from skimage.filters import threshold_otsu as otsu

from skimage.filters import unsharp_mask as unsharp
from skimage.metrics import structural_similarity as ssim
import re

# %%
parent_dir=os.getcwd()
train_dir = os.path.join(parent_dir, 'Project Dataset/Train/')
test_dir = os.path.join(parent_dir, 'Project Dataset/Test/')
out_dir = os.path.join(parent_dir, 'Project Dataset/Output Report/')
template_dir = os.path.join(parent_dir, 'Project Dataset/Templates/')

# %%
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
    #plt.figure()
    #plt.imshow(mask)
    #plt.title('Mask')
    mask = erosion(mask, square(2))     
    #plt.figure()
    #plt.imshow(mask)
    #plt.title('Mask after erosion')
    minutiaeTerm = np.uint8(mask)*minutiaeTerm
    minutiaeBif = np.uint8(mask)*minutiaeBif
    return(minutiaeTerm, minutiaeBif)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# %%


def ShowResults(skel, TermLabel, BifLabel,st=''):
    minutiaeBif = TermLabel * 0
    #print(np.shape(minutiaeBif))
    minutiaeTerm = BifLabel * 0

    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    #DispImg[:, :, 0] = skel;
    DispImg[:, :, 1] = skel
    #DispImg[:, :, 2] = skel;
    
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
    plt.title("Minutiae extraction results "+st)
    plt.imshow(DispImg)

    return minutiaeBif, minutiaeTerm


# %%
def process_image(img_name):
    img = cv2.imread(img_name)
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=cv2.resize(img,(256,256))
    img=img[5:img.shape[0]-5,5:img.shape[1]-5]

    img=unsharp(img,radius=3,amount=2)
    THRESHOLD1=otsu(img)
   

    img = np.array(img > THRESHOLD1).astype(int)
    #invert 1 to 0 and 0 to 1
    img = 1 - img


    
    skel = skimage.morphology.skeletonize(img)
    skel =skel*255

    
    mask = img*255


    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)
    return minutiaeTerm, minutiaeBif,skel

# %%
all_data=[]


train_files=os.listdir(train_dir)
train_files=sorted_alphanumeric(train_files)

test_files=os.listdir(test_dir)
test_files=sorted_alphanumeric(test_files)

n=np.int16(len(train_files)/3)
for i in range(n):
    for j in range(3):
        
        img_name=train_files[i*3+j]
        print(img_name)
        img_name=os.path.join(train_dir,img_name)
        

        minutiaeTerm, minutiaeBif,skel = process_image(img_name)
        (row,col)=minutiaeBif.shape
        
        
        minutiaeBif1=np.reshape(minutiaeBif,(row*col))
        idx=np.where(minutiaeBif1==1)
        data=np.hstack(([1+i],idx[0]))
        #print(data.shape)
        BifLabel = skimage.measure.label(minutiaeBif, connectivity=1)
        TermLabel = skimage.measure.label(minutiaeTerm, connectivity=1)
        all_data.append(data)
    
    









# %%

with open(os.path.join(template_dir, 'template_with_labels.txt'), 'wb') as fp:
    pickle.dump(all_data, fp)

    pickle.dump(all_data, fp)

print("\n\nFIND THE DATABASE TEMPLATE FILE AT:",os.path.join(template_dir, 'template_with_labels.txt'))

# %%



