import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import _split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Function to get the data from XML Annotation
def extract_xml_file(xml_file):
    xml_root = ET.parse(xml_file).getroot()

    # Initialise the info dict 
    img_info_dict = {}
    img_info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in xml_root:
        # Get the file name 
        if elem.tag == "filename":
            img_info_dict['filename'] = elem.text

        # Get size of the image
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            img_info_dict['image_size'] = tuple(image_size)

        # Get bounding box of the image
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            img_info_dict['bboxes'].append(bbox)

    return img_info_dict


class_names = [] # This list container store on all label
class_name_to_id_mapping = {} # This dictionary container mapping all label to unique number.

# get all class names and store on class_name list 
def get_class_names(info_dict):
  for b in info_dict['bboxes']:
      class_names.append(b['class'])

def mapping_to_class_name_to_id(class_names):
  unique_class_names = np.unique(class_names)
  for i, unique_label in enumerate(unique_class_names):
    class_name_to_id_mapping[unique_label] = i


print(extract_xml_file('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/train_labels/Acadian_Flycatcher_0003_29094.xml'))

# Get the all train and validation xml annotations file path
train_annotations_labels = [os.path.join('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/train_labels/', x) for x in os.listdir('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/train_labels/') if x[-3:] == "xml"]
train_annotations_labels.sort()
# test
test_annotations_labels = [os.path.join('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/valid_labels', x) for x in os.listdir('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/valid_labels') if x[-3:] == "xml"] 
test_annotations_labels.sort()

# extract xml file and append label into class_names list container
for i,ann in enumerate(tqdm(train_annotations_labels)):
    info_dict = extract_xml_file(ann)
    get_class_names(info_dict)

# If all label store on list container than mapping them unique number 
mapping_to_class_name_to_id(class_names)


print(len(train_annotations_labels),len(test_annotations_labels),len(class_name_to_id_mapping))