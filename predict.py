#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import urllib
import numpy as np
from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms
from PIL import Image


# In[62]:


def predict(file, file_type="url", category=""): # category可以不給
    
    index2label = {
    0: "abstract",
    1: "animal print",
    2: "camouflage",
    3: "floral",
    4: "geometric",
    5: "ikat",
    6: "melange",
    7: "placement",
    8: "plaids",
    9: "solid",
    10: "spots",
    11: "strips",
    }
    
    def url_to_image_original(url, width=200, height=200):
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    face_cascade = cv2.CascadeClassifier('C:\\Users\\andyy\\anaconda3\\envs\\python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    
    def has_face(image):  # 判斷是否有人臉
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1 , minNeighbors=5, minSize=(int(gray.shape[0]/15),int(gray.shape[1]/20)))
        def get_largest_part(a_list):  # 回傳最大的區塊
            if len(a_list)<2:
                return a_list
            else:
                return sorted(a_list, key=lambda x: x[2]*x[3], reverse=True)[:1]
        faces = get_largest_part(faces)
        if faces == ():
            return False  # 如果沒有偵測到，或是偵測出來位置太低就回傳否
        elif faces[0][1] > image.shape[0] * 0.3:
            return False
        else:
            return True
        
    def cut_image(cur_image, original_clothing):  # 切割衣服
        upper_clothing = ["Short Sleeve", "Softshells", "Long Sleeve", "Windwear", "Hoodie & Sweatshirt", "Jackets & Vests",
                         "Tank", "Tops", "Sports Bra", "Polo", "Vest", "Insulated & Down", "Fleece", "Shirt"]
        lower_clothing = ["Shorts","Pants", "Leggings & Tights", "Bottoms"]
        lower_all_clothing = ["Skirt & Dresses"]
        upper_all_clothing = ["Surfing", "Rainwear", "WetSuit", "Weather Protection", "Springsuit"]
        if bool(set(original_clothing) & set(upper_clothing)):  #上半身情況
            if has_face(cur_image):
                y_start_coefficient = 0.3
                height_coefficient = 0.5
            else:
                y_start_coefficient = 0.2
                height_coefficient = 0.6
            clothing_condition = 0
        elif bool(set(original_clothing) & set(lower_clothing)):  #下半身情況
            if has_face(cur_image):
                y_start_coefficient = 0.4
                height_coefficient = 0.5
            else:
                y_start_coefficient = 0.15
                height_coefficient = 0.65
            clothing_condition = 1
        elif bool(set(original_clothing) & set(lower_all_clothing)):  #下半身全身都有可能情況
            if has_face(cur_image):
                y_start_coefficient = 0.25
                height_coefficient = 0.55
            else:
                y_start_coefficient = 0.2
                height_coefficient = 0.6
            clothing_condition = 2
        elif bool(set(original_clothing) & set(upper_all_clothing)):  #上半身全身都有可能情況
            if has_face(cur_image):
                y_start_coefficient = 0.3
                height_coefficient = 0.5
            else:
                y_start_coefficient = 0.2
                height_coefficient = 0.6
            clothing_condition = 3
        else:
            y_start_coefficient = 0.2
            height_coefficient = 0.6
            clothing_condition = 4
        
        x_start = int(0.2 * cur_image.shape[1])
        y_start = int(y_start_coefficient * cur_image.shape[0])
        # 裁切區域的長度與寬度
        width = int(0.6 * cur_image.shape[1])
        height = int(height_coefficient * cur_image.shape[0])
        # 裁切圖片
        crop_img = cur_image[y_start:y_start+height, x_start:x_start+width]
        # 裁切完再resize
        train_images_cur = cv2.resize(crop_img, (256, 256))
        return train_images_cur, clothing_condition



    
    if file_type == "url":
        image = url_to_image_original(file)
    elif file_type == "file_name":
        image = Image.open(file)
        image = np.array(image)
    else:
        print("Wrong file type.")
        return None
        
    img_ = np.array(image)
    if img_.ndim != 3:
        print("dimension error")
        return None
    if img_.shape[2] == 4:  # 處理有 Transparency (RGBA) 的情況
        img = img.convert('RGB')
        img_ = np.array(img)
    image, clothing_condition = cut_image(img_, category) 
    
#     import matplotlib.pyplot as plt 
#     print(image.shape)
#     plt.imshow(image)
#     plt.show()
    
    
    datagen_valid_test = transforms.Compose([
    transforms.ToTensor(),
    ])
    image = datagen_valid_test(Image.fromarray(image))
    image = torch.unsqueeze(image, 0)
    
    model = EfficientNet.from_name('efficientnet-b1')
    model.load_state_dict(torch.load('./models/efficientnet.h5', map_location={'cuda:0': 'cpu'}))
    with torch.no_grad():
        model.eval()
        inputs = image.float()  # 因為train有經過資料增強，已是float型態，valid、test也必須轉成float型態
        outputs = model(inputs)
        outputs = torch.argmax(outputs, 1)
        
        return index2label[outputs.item()]
    
    
    
print("example: url + category")    
output = predict("https://bilab.synology.me/smis/brand/reebok/reebok_kxxojml3.png", "url", "Hoodie & Sweatshirt")
print(output)
print("example: url + without category")    
output = predict("https://bilab.synology.me/smis/brand/reebok/reebok_kxxojml3.png", "url")
print(output)
print("example: file + category")
output = predict("./batch1_images_cut_formal/train/abstract/adidas__hjvbsam.png", "file_name", "Hoodie & Sweatshirt")
print(output)
print("example: file + without category")
output = predict("./batch1_images_cut_formal/train/abstract/adidas__hjvbsam.png", "file_name")
print(output)




