# %%
import numpy as np
import os
import pickle
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, erosion, square

import skimage
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
template_file=os.path.join(template_dir,'template_with_labels.txt')
with open(template_file, 'rb') as fp:
    all_data=pickle.load(fp)



# %%
train_files=os.listdir(train_dir)
train_files=sorted_alphanumeric(train_files)

test_files=os.listdir(test_dir)
test_files=sorted_alphanumeric(test_files)



# %%
while 1:
    msg="Type 'all' to predict entire Data. Or Enter a number between 1 and "+str(len(test_files))+' to predict a single image. To Quit: Press ctrl+c\n'
    val=input(msg)

          

    if val.lower()=='all':
        outfile=os.path.join(out_dir,'testresult.txt')
        #print(outfile)

        y_true=np.zeros(len(test_files))
        y_pred=np.zeros(len(test_files))
        for i in range(40):
            y_true[i]=i+1

        file=open(outfile,'wt')
        for f in range(len(test_files)):
            #img_name=test_files[int(val)-1]
            img_name=test_files[f]


            img_full_name=test_dir+img_name
            #print(img_full_name)
            minutiaeTerm, minutiaeBif,skel=process_image(img_full_name)
            (row,col)=minutiaeBif.shape
            mbvect=np.reshape(minutiaeBif,(row*col))
            s_all=np.zeros(len(train_files))
            for i in range(len(train_files)):
                data=all_data[i]
                idx=data[1:]

                
                temp=np.zeros((len(mbvect)))
                temp[idx]=1
                temp=np.reshape(temp,(row,col))
                
                
                s=ssim(minutiaeBif,temp)
                s_all[i]=s
            pred_idx=np.argmax(s_all)
            

            pred_img=train_files[pred_idx]
            
            if s_all[pred_idx]>=0.7:
                file.writelines("***Image Selected: "+img_name+"  Predicted Image: "+pred_img+" Highest SSIM: "+str(np.round(s_all[pred_idx],3))+"\n")
                y_pred[f]=all_data[pred_idx][0]
            else:
                file.writelines("***Image Selected: "+img_name+"  Predicted Image: NOT IN DATABASE"+" Highest SSIM: "+str(np.round(s_all[pred_idx],3))+"\n")
                y_pred[f]=0
            #print(line)
            #f.write(line)

            
        file.close()
        print("\n \n***SEE OUTPUT FILE FOR ENTIRE DATA AT  ",outfile," for results")
        
        correct_pred=len(np.where(y_true==y_pred)[0])
        print("Correct Predictions: ",correct_pred, " out of ",len(test_files))
        break
    elif val.isdigit():
        if int(val)>0 and int(val)<=len(test_files):
            img_name=test_files[int(val)-1]
            #img_name=test_files[f]


            img_full_name=test_dir+img_name
            #print(img_full_name)
            minutiaeTerm, minutiaeBif,skel=process_image(img_full_name)
            (row,col)=minutiaeBif.shape
            mbvect=np.reshape(minutiaeBif,(row*col))
            s_all=np.zeros(len(train_files))
            for i in range(len(train_files)):
                data=all_data[i]
                idx=data[1:]

                
                temp=np.zeros((len(mbvect)))
                temp[idx]=1
                temp=np.reshape(temp,(row,col))
                
                
                s=ssim(minutiaeBif,temp)
                s_all[i]=s
            pred_idx=np.argmax(s_all)
            

            pred_img=train_files[pred_idx]
            
            if s_all[pred_idx]>0.65:
                print("***Image Selected: "+img_name+"  Predicted Image: "+pred_img+" Highest SSIM: "+str(np.round(s_all[pred_idx],3)))
                
            else:
                print("***Image Selected: "+img_name+"  Predicted Image: NOT IN DATABASE"+" Highest SSIM: "+str(np.round(s_all[pred_idx],3)))

        else:
            print("Invalid Input")
            continue
    else:
        print("Invalid Input")
        continue

        


                


