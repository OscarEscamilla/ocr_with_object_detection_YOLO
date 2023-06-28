# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
import cv2
model = YOLO('./best.pt')

img = cv2.imread('./data/images/train/01_2309035106870.jpg')
  # load a pretrained YOLOv8n detection model  # train the model
results = model(img)  # predict on an image

# Visualize the results on the frame
annotated_frame = results[0].plot()



# Display the annotated frame
cv2.imshow("YOLOv8 Inference", annotated_frame)

cv2.waitKey(0)


#yolo task=detect mode=predict model=custom_ocr_model.pt show=True

#./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74