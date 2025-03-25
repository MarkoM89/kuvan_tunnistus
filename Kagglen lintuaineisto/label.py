from lintujen_tunnistus import *

#Convert the info dict to the required yolo txt file format and write it to disk
def convert_to_yolov8(info_dict,path):
    print_buffer = []

    # For each bounding box
    for bbox in info_dict["bboxes"]:


        try:
            # get class id for each label
            class_id = class_name_to_id_mapping[bbox["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v8
        b_center_x = (bbox["xmin"] + bbox["xmax"]) / 2 
        b_center_y = (bbox["ymin"] + bbox["ymax"]) / 2
        b_width    = (bbox["xmax"] - bbox["xmin"])
        b_height   = (bbox["ymax"] - bbox["ymin"])

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 

        #Write the bounding box details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save same as image file name.
    save_file_name = os.path.join(path, info_dict["filename"].replace("jpg", ""))
    save_file_name += '.txt'
    print(save_file_name)
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))


    # Convert and save the train annotations
for i,ann in enumerate(tqdm(train_annotations_labels)):
    info_dict = extract_xml_file(ann)
    convert_to_yolov8(info_dict,'C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/train_images/')

annotations_labels = [os.path.join('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/train_images/', x) for x in os.listdir('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/train_images/') if x[-3:] == "txt"]

# Convert and save the test annotations
for i,ann in enumerate(tqdm(test_annotations_labels)):
    info_dict = extract_xml_file(ann)
    convert_to_yolov8(info_dict,'C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/valid_images/')

test_annotations_labels = [os.path.join('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/valid_images/', x) for x in os.listdir('C:/Users/Marko/Desktop/metropolia koulu/Opinnäytetyö/CUB 200 Bird Species XML Detection Dataset/cub_200_2011_xml/valid_images/') if x[-3:] == "txt"]


print(len(train_annotations_labels),len(test_annotations_labels))



