import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import sys
import logging
import datetime
from db_helper import db_helper 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parking_detection.log'),
        logging.StreamHandler()
    ]
)

# Path configuration
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT
MODEL_PATH = DATASET_DIR / "outputs" / "parking_model.pt"
TRAINED_MODEL = DATASET_DIR / "outputs" / "trained_parking_model.pt"
PARKING_ROOT = PROJECT_ROOT.parent.parent / "xmp" / "htdocs" / "ParkEasy-main"
PARKING_IMAGES = PARKING_ROOT / "application" / "parking_images"

# Create directories if they don't exist
PARKING_IMAGES.mkdir(parents=True, exist_ok=True)
(DATASET_DIR / "outputs").mkdir(parents=True, exist_ok=True)

def train_model():
    """Train a YOLOv8 model on the parking space dataset"""
    try:
        logging.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        # Verify dataset structure
        yaml_path = DATASET_DIR / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

        # Load a pretrained YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Fix paths in data.yaml
        corrected_yaml = DATASET_DIR / "corrected_data.yaml"
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()

        yaml_content = yaml_content.replace(
            "train: ../train/images", 
            f"train: {(DATASET_DIR / 'train' / 'images').as_posix()}"
        )
        yaml_content = yaml_content.replace(
            "val: ../valid/images", 
            f"val: {(DATASET_DIR / 'valid' / 'images').as_posix()}"
        )
        yaml_content = yaml_content.replace(
            "test: ../test/images", 
            f"test: {(DATASET_DIR / 'test' / 'images').as_posix()}"
        )

        with open(corrected_yaml, 'w') as f:
            f.write(yaml_content)

        # Train the model
        results = model.train(
            data=corrected_yaml.as_posix(),
            epochs=10,
            imgsz=512,
            batch=8,
            name="parking_space_detection",
            project=(DATASET_DIR / "outputs").as_posix(),
            exist_ok=True
        )


        model.save(TRAINED_MODEL.as_posix())
        logging.info(f"Model training complete. Saved to: {TRAINED_MODEL}")
        os.remove(corrected_yaml)
        return model

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        return None

def detect_parking_spaces(model, image_path, confidence=0.5, location="default"):
    """Detect parking spaces in a new image and update database"""
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = cv2.imread(image_path.as_posix())
        if img is None:
            raise ValueError("Could not read image file")

        results = model.predict(img, conf=confidence)

        empty = occupied = 0
        detection_boxes = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                coords = box.xyxy[0].tolist()

                if class_id == 0:
                    empty += 1
                else:
                    occupied += 1

                detection_boxes.append({
                    'class': 'empty' if class_id == 0 else 'occupied',
                    'confidence': confidence,
                    'coordinates': coords
                })

        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_image_filename = f"parking_{timestamp}.jpg"
        processed_image_path = PARKING_IMAGES / processed_image_filename

        # Save the processed image
        visualize_results(image_path, {
            'status': 'success',
            'detections': detection_boxes
        }, output_path=processed_image_path.as_posix())

        # Update database with all information
        db_helper.update_parking_status(
            available=empty,
            occupied=occupied,
            location=location,
            image_path=f"parking_images/{processed_image_filename}"  # Web-accessible path
        )

        return {
            'status': 'success',
            'available': empty,
            'occupied': occupied,
            'total_spaces': empty + occupied,
            'image_size': f"{img.shape[1]}x{img.shape[0]}",
            'timestamp': datetime.datetime.now().isoformat(),
            'detections': detection_boxes,
            'image_path': processed_image_path.as_posix(),
            'location': location
        }

    except Exception as e:
        logging.error(f"Detection error: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.datetime.now().isoformat(),
            'location': location
        }

def visualize_results(image_path, results, output_path=None):
    """Visualize detection results on the image"""
    try:
        if hasattr(image_path, 'as_posix'):
            image_path = image_path.as_posix()
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")

        colors = {'empty': (0, 255, 0), 'occupied': (0, 0, 255)}
        
        for detection in results.get('detections', []):
            coords = detection['coordinates']
            cv2.rectangle(img, 
                         (int(coords[0]), int(coords[1])),
                         (int(coords[2]), int(coords[3])),
                         colors[detection['class']], 2)
            
            label = f"{detection['class']} {detection['confidence']:.2f}"
            cv2.putText(img, label,
                       (int(coords[0]), int(coords[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[detection['class']], 1)

        if output_path:
            if hasattr(output_path, 'as_posix'):
                output_path = output_path.as_posix()
            cv2.imwrite(output_path, img)
            logging.info(f"Results saved to: {output_path}")
        
        return img

    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Parking Space Detection System')
    parser.add_argument('--image', type=str, help='Path to parking lot image')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--location', type=str, default="default", help='Parking location identifier')
    args = parser.parse_args()

    try:
        # Load or train model
        if os.path.exists(TRAINED_MODEL) and not args.train:
            logging.info("Loading existing model...")
            model = YOLO(TRAINED_MODEL.as_posix())
        else:
            logging.info("Training new model...")
            model = train_model()
            if model is None:
                return 1

        # Handle image detection
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                logging.error(f"Image not found: {image_path}")
                return 1
                
            result = detect_parking_spaces(model, image_path, location=args.location)
            
            if args.visualize and result['status'] == 'success':
                output_img = visualize_results(
                    image_path, 
                    result,
                    output_path=(DATASET_DIR / "outputs" / "detection_result.jpg").as_posix()
                )
                if output_img is not None:
                    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
            
            print(json.dumps(result, indent=2))
            return 0
            
        else:
            # Default test image
            test_image = DATASET_DIR / "test" / "images" / "135-2319_jpg.rf.8200d560687e3acbe58fdb921d0642fd.jpg"
            if test_image.exists():
                result = detect_parking_spaces(model, test_image, location=args.location)
                
                if args.visualize and result['status'] == 'success':
                    output_img = visualize_results(
                        test_image, 
                        result,
                        output_path=(DATASET_DIR / "outputs" / "detection_result.jpg").as_posix()
                    )
                    if output_img is not None:
                        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
                        plt.axis('off')
                        plt.show()
                
                print(json.dumps(result, indent=2))
                return 0
            else:
                logging.error("Test image not found")
                return 1

    except Exception as e:
        error_data = {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(json.dumps(error_data, indent=2))
        return 1
    finally:
        db_helper.close()

if __name__ == "__main__":
    try:
        from ultralytics import YOLO
        sys.exit(main())
    except ImportError:
        logging.critical("Required packages missing. Please install with:")
        print("pip install ultralytics opencv-python matplotlib numpy mysql-connector-python")
        sys.exit(1)