#PROJECT PLASTIC AI
#G17 University Ambassadors Consortium
#University of Moratuwa
#SDG Goal 14 - Life below water
#P M Ranasinghe
#11/10/2021
#%%

#Exploring the dataset

#Importing the required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from tqdm import tqdm   

#Reading the images
path="Data_Set"
data=[os.path.join(path,"train",x) for x in os.listdir(os.path.join(path,"train"))]

images = [data[x] for x in range(len(data)) if data[x].endswith('.jpg')]
files = [data[x] for x in range(len(data)) if data[x].endswith('.txt')]

print(f"There are {len(images)} images in the training set")

##checking a random Image
random_num=random.randint(0, len(images))
plt.imshow(cv2.cvtColor(cv2.imread(images[random_num]),cv2.COLOR_BGR2RGB))

#%% Exploring the txt files and making a dataframe 
df = pd.DataFrame(columns=['image_name','object','x_cen','y_cen','w','h'])
i=0
k=0
for id in tqdm(images):
    label = open(id[:-4]+'.txt','r') 
    a = label.read() 
    label.close()
    data_txt = a.split('\n')
    #cleaned_label = data_txt[:-1]
    cleaned_label = data_txt
    for name in cleaned_label:
        ind_label = name.split(' ')
        df.loc[i,'image_name'] = id[15:]
        df.loc[i,'object'] = int(ind_label[0])
        df.loc[i,'x_cen'] = float(ind_label[1])
        df.loc[i,'y_cen'] = float(ind_label[2])
        df.loc[i,'w'] = float(ind_label[3])
        df.loc[i,'h'] = float(ind_label[4])
        i=i+1

print(df) 
       
#%%Converting to Pascal VOC format

#column name for pascal-voc dataframe
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
pascal_voc = pd.DataFrame(columns=column_name)

for i in tqdm(range(len(df))):
    
    pascal_voc.loc[i,'filename'] = df.loc[i,'image_name']
    pascal_voc.loc[i,'width'] = 416
    pascal_voc.loc[i,'height']= 416  
    if df.loc[i,'object'] == 0:    
        pascal_voc.loc[i,'class'] = 'Plastic'
    else:
        pascal_voc.loc[i,'class'] = 'Not a Plastic'
    pascal_voc.loc[i,'xmin']  = int((df.loc[i,'x_cen'] - df.loc[i,'w']/2)*416)
    pascal_voc.loc[i,'ymin']  = int((df.loc[i,'y_cen'] - df.loc[i,'h']/2)*416)
    pascal_voc.loc[i,'xmax']  = int((df.loc[i,'x_cen'] + df.loc[i,'w']/2)*416)
    pascal_voc.loc[i,'ymax']  = int((df.loc[i,'y_cen'] + df.loc[i,'h']/2)*416) 

print(pascal_voc.head())

#%%Function to Output random image with bounding boxes

def show_samples(any_index_number = random.randint(0, len(df))):
    any_index_number = random.randint(0, len(df)) #change this number for different images
    image_name=path+"/train/"+pascal_voc.loc[any_index_number,'filename']
    test_img = cv2.cvtColor(cv2.imread(image_name),cv2.COLOR_BGR2RGB)
    test_df = pascal_voc[pascal_voc['filename']==pascal_voc.loc[any_index_number,'filename']].reset_index(drop=True)
    
    for i in range(len(test_df)):
        cv2.rectangle(test_img, (test_df.loc[i,'xmin'],test_df.loc[i,'ymin']), (test_df.loc[i,'xmax'],test_df.loc[i,'ymax']),(12,204,204), 2)
        test_img[test_df.loc[i,'ymin']-23:test_df.loc[i,'ymin'],test_df.loc[i,'xmin']-1:test_df.loc[i,'xmin']+85]=(12,204,204)
        cv2.putText(test_img, test_df.loc[i,'class'], (test_df.loc[i,'xmin'],test_df.loc[i,'ymin']-3), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    return test_img

test_img=show_samples()
plt.figure(figsize=(8,8))    
plt.imshow(test_img)
plt.axis('off')
plt.show()

#Making a figure containing 16 images with bounding boxes
lst=random.sample(range(0,len(df)), 16)
plt.figure(figsize=(15,15))
for index,value in  enumerate(lst):
    img_subplot=show_samples(value)
    plt.subplot(4,4,index+1)
    plt.imshow(img_subplot)
    plt.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.savefig(f"Images/{lst[0]}.png")
plt.show()  

#%%How many Bounding boxes 
box_count=pascal_voc["filename"].value_counts()
plt.figure(figsize=(8,8))
plt.hist(box_count) 
plt.title("Bounding Box Count Distribution") 
plt.xlabel("No of Boxes")
plt.ylabel("Count")  
plt.show()
print(box_count.value_counts())
