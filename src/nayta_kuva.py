from lintujen_tunnistus import *


random.seed(0)

# Reverse order by class names. example is: 0 : bird_name. 

class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_image_with_bounding_box(image, annotation_list):
  '''
     image : It's actual numpy formatted image you input.
     annotation_list : It's give as label with bounding box.

  '''
  # conver numpy array
  annotations = np.array(annotation_list)
  # get image width and height and store them different variable
  w, h = image.size

  plotted_image = ImageDraw.Draw(image)

  t_annotations = np.copy(annotations)
  t_annotations[:,[1,3]] = annotations[:,[1,3]] * w
  t_annotations[:,[2,4]] = annotations[:,[2,4]] * h 

  t_annotations[:,1] = t_annotations[:,1] - (t_annotations[:,3] / 2)
  t_annotations[:,2] = t_annotations[:,2] - (t_annotations[:,4] / 2)
  t_annotations[:,3] = t_annotations[:,1] + t_annotations[:,3]
  t_annotations[:,4] = t_annotations[:,2] + t_annotations[:,4]

  for ann in t_annotations:
      obj_cls, x0, y0, x1, y1 = ann
      plotted_image.rectangle(((x0,y0), (x1,y1)))

      plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

  plt.imshow(np.array(image))
  plt.show()



  # Get any random label file 
label_file = random.choice(train_annotations_labels)
with open(label_file, "r") as file:
    label_with_bounding_box = file.read().split("\n")[:-1]
    label_with_bounding_box = [x.split(" ") for x in label_with_bounding_box]
    label_with_bounding_box = [[float(y) for y in x ] for x in label_with_bounding_box]

# Get the equal image file
image_file = label_file.replace("annotations", "images").replace("txt", "jpg")

assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)


# Plot the Bounding Box
plot_image_with_bounding_box(image, label_with_bounding_box)



  