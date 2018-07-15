import os
import cv2
import numpy as np

def get_dir_images(dir_name):
    img_names = [fname for fname in os.listdir(dir_name) 
                 if fname.lower().endswith(".png") or fname.lower().endswith(".jpg")]

    img_names.sort()
    
    return img_names

def load_image(path):
    image = cv2.imread(path)
    image = np.expand_dims(image,axis=0)
    return image

def save_image(path,img):
    cv2.imwrite(path,img)
    

def apply_mask(image,mask,alpha=0.7):
    """Apply the given mask to the image.
    """
    #for c in range(3):
    image[:, :, 0] = np.where(mask >= 0.5,
                              image[:, :, 0] *
                              (1 - alpha) + alpha * 0.7 * 255,
                              image[:, :, 0])
    return image


def draw_bounding_box(img,detections,boxes,classes,class_map,masks=None):
    img_height,img_width,_ = img.shape
    
    for i in detections:
        #Draw bounding box 
        ymin, xmin, ymax, xmax = boxes[i]
        
        xmin = int(xmin*img_width)
        xmax = int(xmax*img_width)
        ymin = int(ymin*img_height)
        ymax = int(ymax*img_height)
        


        #Rectangle thickness
        box_thickness = int(img_width/1000.0)
        font_scale = int(img_width/1000.0)
        font_thickness = 2

        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),thickness = box_thickness)
        
        box_height = ymax - ymin
        box_width = xmax - xmin

        if masks is not None:

            #normalize mask_values
            masks[i] *= 1.0/masks[i].max()

            # #Quantize
            # masks[i] = np.where()
            
            #scale mask
            scaled_mask = cv2.resize(masks[i],(box_width,box_height))

            #mask as image
            image_mask = np.zeros((img_height,img_width))
            
            image_mask[ymin:ymax,xmin:xmax] = scaled_mask

            #apply mask
            img = apply_mask(img,image_mask)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        size = cv2.getTextSize(class_map[int(classes[i])],cv2.FONT_HERSHEY_SIMPLEX,font_scale,font_thickness)

        text_width,text_height = size[0]
        
        #Background
        cv2.rectangle(img,(xmin,ymin+text_height+1),(xmin+text_width+1,ymin),(0,0,255),thickness = -1)
        
        #Foreground Text
        cv2.putText(img,class_map[int(classes[i])],(xmin,ymin+text_height), font, font_scale,(255,255,255),font_thickness,cv2.LINE_AA)


def get_class_map(class_map_file):
    with open(class_map_file,'r') as csv_file:
        class_id_list = csv_file.readlines()

    class_map = {}
    for id_name in class_id_list:
        id_,name = id_name.strip().split(",")
        class_map[int(id_)] = name
    
    return class_map

def get_detections(scores,threshold_score):
    detections = []
    
    for i,score in enumerate(scores):
        if score >= threshold_score:
            detections.append(i)

    return detections
