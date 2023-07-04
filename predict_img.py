# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
import cv2
import re
from pprint import pprint
import numpy as np
from PIL import Image
import pytesseract

class YOLOSegmentation:

    def __init__ (self):
        self.model = YOLO("./best.pt")


    def detect(self, img) :
        # Get img shape
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
      
        bboxe = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_id = np.array(result.boxes.cls.cpu(), dtype="int")
         # Get scores
        score = np.array(result.boxes.conf.cpu(), dtype="float")
        # Get Label
        #label = result.names[class_id]

        final_data = []

        for i in range(0,len(class_id)):
          id = class_id[i]
          final_data.append({"id":id,"label": result.names[id], "score": score[i].round(2),"boxes": bboxe[i]})

      
        return final_data, result
    
    def apply_filter(self, img):
      # Grayscale, Gaussian blur, Otsu's threshold
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (3,3), 0)
      thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

      # Morph open to remove noise and invert image
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
      invert = 255 - opening

      return gray
    

class TextUtils:

 

  def extract_text(self, img): 
    custom_config = r'-l eng --oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text

  def clear_date(self, txt):
    return re.sub("[a-zA-Z]","", txt)
    

img = cv2.imread('./dataset/images/train/01_2309015110552.jpg')
yolo = YOLOSegmentation()
utils = TextUtils()
final_data, result = yolo.detect(img)

for item in final_data:
  x,y,x2,y2 = item["boxes"]
  #cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
  img_crop = img[y:y2, x:x2]
  img_filtered = yolo.apply_filter(img_crop)
  text = utils.extract_text(img_filtered)

  if item["label"] == "antiguedad" or "expedicion" or "vigencia":
    print(utils.clear_date(text))
  else:
    print(text)
  
  cv2.imshow("YOLOv8 Inference", img_filtered)
  cv2.waitKey(0)












