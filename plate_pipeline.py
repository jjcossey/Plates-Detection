import torch
import cv2
import easyocr
import os

class LicensePlateRecognizer:
    """
    A pipeline that loads a YOLOv5 trained model for license plate detection 
    and uses EasyOCR to perform OCR on the detected plate region.
    """
    def __init__(self, weight_path, device='cpu'):
        self.device = device
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path=weight_path, device=device, force_reload=True)
        print("Initializing EasyOCR Reader...")
        self.reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
    
    def detect_plate(self, img):
        results = self.model(img)
        detections = results.xyxy[0].cpu().numpy()
        if len(detections) == 0:
            return None, detections
        # For simplicity, select the first detection.
        x_min, y_min, x_max, y_max, conf, cls = detections[0]
        plate_roi = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        return plate_roi, detections
    
    def recognize_plate(self, plate_roi):
        # Convert ROI to RGB as EasyOCR expects an RGB image.
        if len(plate_roi.shape) == 3:
            plate_rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
        else:
            plate_rgb = plate_roi
        
        # Run EasyOCR on the cropped plate region.
        results = self.reader.readtext(plate_rgb)
        extracted_text = ' '.join([res[1] for res in results])
        return extracted_text.strip()
    
    def predict_plate(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to load image")
        plate_roi, detections = self.detect_plate(img)
        if plate_roi is None:
            return None, detections, img
        plate_text = self.recognize_plate(plate_roi)
        return plate_text, detections, img

# Example usage:
if __name__ == "__main__":
    weight_file = r'C:\Users\user\Desktop\yolov5\yolov5\runs\train\exp5\weights\best.pt'
    test_image_file = r'C:\Users\user\Desktop\VOC-LICENSE-PLATE\JPEGImages\Cars1.png'
    
    recognizer = LicensePlateRecognizer(weight_file, device='cpu')
    plate_text, detections, original_img = recognizer.predict_plate(test_image_file)
    
    if plate_text:
        print("Detected License Plate Text:", plate_text)
    else:
        print("No license plate detected.")
