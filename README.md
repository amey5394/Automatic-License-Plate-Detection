# Automatic License Plate Detection 
## Object detection and Image classification problem

License plate systems play an important role in multiple fields such as for automatic toll payment, wherein the car will be automatically identified via its license plate to carry out a payment in an automated environment. Parking fee payment and traffic surveillance are some other examples which heavily relies on License plate numbers. The automated nature of license plate systems makes it a viable and cost-effective method to be implemented.

Object detection is carried out with the help of Faster-RCNN. The model was used to detect the number of plates and identify the characters inside the number plate. The report will focus on as follows: Section 2 will focus on the Dataset followed by Section 3 which will focus on Pre-processing the dataset. Section 4 focuses on all the models and respective steps that have been used to process the license plate while detecting the plate and characters. This section will present you with both success and failure models during the experiments. 

### Dataset
The dataset for this project was acquired from University of Zagreb, Licence plate detection, recognition, and automated storage. The dataset consists a total of 510 images of motor vehicle number plates. The dataset was further split to 80-20 for train-test purpose. Below you can see sample images present in the dataset.
![image](https://user-images.githubusercontent.com/30070656/91628817-dc015680-ea06-11ea-9a56-9bd28d3fdb55.png)

### Workflow
![image](https://user-images.githubusercontent.com/30070656/91628842-04895080-ea07-11ea-8c2a-951a4e9272e2.png)

## Model-1 (Locating the license plate)
### Data Preprocessing
Pre-processing:
As different approaches will be applied to both the models in different stages, we have preprocessed our images and generated supported files accordingly. The images that we had received from the source were of different sizes, example: 640*480, 1024*768 and 1600*1200. We resized all the images to 640*480 first and carried out the annotation process with all the images. We have used VGG Image Annotator online tool for the annotation of our images. The annotation details were then received in csv file with respective values as shown in the figure below:

![image](https://user-images.githubusercontent.com/30070656/91628940-06074880-ea08-11ea-8782-b426fbee9714.png)

Next sub step, to generate csv and pvtxt file in the format our model required, we used the script from xml_to_csv.py (Github resource file) and modified as per our requirement. This generated the csv file in the following format:
![image](https://user-images.githubusercontent.com/30070656/91628982-5e3e4a80-ea08-11ea-9acd-628e85f3ad7d.png)

and the generated .pbtxt file consists of only one item with the class name “License_Plate”. After this with the help of generate_tfrecord.py file using generated csv and .pbtxt file we generated .record file. By this phase, we had all the supportive files that we require to to train
our model. During the first stage, we had worked to identify the location of the number plate in the image itself. We had carried out this experiment with two standard models: SSD and Faster-RCNN. Following are the experimental settings and respective results for each of the standard model:

## Architecture Used
### Experiment - 1 (Using Single Shot Detector - SSD)
With our previous work and projects, we learned that SSD is quite faster in comparison to other models so first, we tried to work with the SSD model. Experimental settings: The configuration and pipeline we used for the SSD model is as follows: 
```python
'ssd_mobilenet_v2': {
    'model_name' : 'ssd_mobilenet_v2_coco_2018_03_29',
    'pipeline_file' : 'ssd_mobilenet_v2_coco.config',
    'batch_size': 4
    }
```
num_steps = 1500 and num_eval_steps = 10
Results: As a result, we have received the images with multiple bounding box with higher accuracy. Some sample images are as follows:
** Note these images below are with the anchor box
![image](https://user-images.githubusercontent.com/30070656/91629103-3e5b5680-ea09-11ea-9a18-af27c5a3488e.png)

With the total of 1500 steps total loss and end learning rate are as follows:
Loss/total_loss = 13.958896, global_step = 1500, learning_rate = 0.004, loss = 13.958896

### Experiment - 2 (Using Faster - RCNN)
While working with Faster-RCNN, we had the same pre-processing steps and use same annotations with same csv, .pbtxt and record files and implemented Faster-RCNN configuration as per the requirement of the standard model.
Experimental Settings for Faster-RCNN:
```python
'faster_rcnn_inception_v2': {
      'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
      'pipeline_file': 'faster_rcnn_inception_v2_coco.config',
      'batch_size': 8
      }
```
with num_steps = 1200 and num_eval_steps = 16;
Results: With this configuration and training Faster-RCNN gave us quite clear result. The images from the experiments are as follows:
![image](https://user-images.githubusercontent.com/30070656/91629191-20422600-ea0a-11ea-9dbe-79de453a8b03.png)

with this model the average precision and average recall received are as follows:
![image](https://user-images.githubusercontent.com/30070656/91629205-3a7c0400-ea0a-11ea-9c95-f5e4d9767fcb.png)

As we received better results from faster-RCNN we used the model for our first phase to detect the license plate. For the second phase of work, only that license plate is the area of our interest, so instead of processing all the images, we cropped the images only including our area of interest and planned to proceed with that. As we can see (images given below) the section of the number plate is not clear so to decrease the noise we converted the images to bitmap, thus the image will include either 0 or 1(255) no intermediate pixel value. In addition to that, the area of our interest (the license plate) is in the image varied in size, so we resized the cropped image to 200*90.

![image](https://user-images.githubusercontent.com/30070656/91629245-a494a900-ea0a-11ea-8edb-d48e2842ba98.png)

The output of Model 1: While predicting the number plate if the predicting image(on left) is as follows then the output from model one to feed to model2 will be the image on right
![image](https://user-images.githubusercontent.com/30070656/91629298-13720200-ea0b-11ea-91fc-ccd81677bd7c.png)

The part of the code that performs this crop, conversion to bit image and resizes it is given below:
```python
	def crop_images(self,img_path,output_dict):
		inx = list(output_dict['detection_scores']).index(max(output_dict['detection_scores']))
		bounding_box = output_dict['detection_boxes'][inx]
		image_save_to = path_to_cropped_images +"/"+ img_path.split("/")[-1]
		image_sel = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
		height,width = image_sel.shape
		y1=int(bounding_box[0]*height)
		x1=int(bounding_box[1]*width)
		y2=int(bounding_box[2]*height)
		x2=int(bounding_box[3]*width)
		new_img = cv2.resize(image_sel[y1:y2,x1:x2],(200,90))
		(thresh, im_bw) = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		cv2.imwrite(image_save_to,im_bw)
		return image_save_to;
```
