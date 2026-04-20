import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = os.getcwd()  # Uses current working directory
SAMPLE_IMAGE = "135-2319_jpg.rf.8200d560687e3acbe58fdb921d0642fd.jpg"
OUTPUT_DIR = "outputs"  # Directory for saving visualizations

def visualize_parking_polygons(image_path, label_path):
    """Visualize parking polygons with enhanced visualization features"""
    try:
        # Validate paths
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Parse polygon annotations
        annotations = []
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 3 or (len(parts)-1) % 2 != 0:
                        print(f"Warning: Skipping line {line_num} - invalid format")
                        continue
                    
                    class_id = int(parts[0])
                    coords = parts[1:]
                    polygon = [(int(coords[i]*width), int(coords[i+1]*height)) 
                             for i in range(0, len(coords), 2)]
                    
                    # Calculate polygon area for visualization
                    area = cv2.contourArea(np.array(polygon))
                    annotations.append({
                        'class_id': class_id,
                        'polygon': polygon,
                        'area': area
                    })
                    
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")

        # Visualization settings
        colors = {
            0: (0, 255, 0),  # Green for empty
            1: (255, 0, 0)    # Red for occupied
        }
        labels = {
            0: "Empty",
            1: "Occupied"
        }
        
        # Create a transparent overlay for better visualization
        overlay = img.copy()
        alpha = 0.3  # Transparency factor

        # Draw each polygon
        for ann in annotations:
            class_id = ann['class_id']
            polygon = ann['polygon']
            
            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [np.array(polygon)], colors[class_id])
            
            # Draw polygon outline on main image
            cv2.polylines(img, [np.array(polygon)], True, colors[class_id], 2)
            
            # Add label at first vertex
            x, y = polygon[0]
            cv2.putText(img, f"{labels[class_id]} ({ann['area']:.0f}px)", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 1, cv2.LINE_AA)

        # Combine overlay with original image
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Create figure with statistics
        plt.figure(figsize=(16, 10))
        plt.imshow(img)
        plt.axis('off')
        
        # Add title with statistics
        empty_count = sum(1 for ann in annotations if ann['class_id'] == 0)
        occupied_count = sum(1 for ann in annotations if ann['class_id'] == 1)
        plt.title(f"Parking Space Analysis\nEmpty: {empty_count} | Occupied: {occupied_count} | Total: {len(annotations)}",
                 pad=20, fontsize=12)

        # Save and display
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = f"polygon_{os.path.splitext(SAMPLE_IMAGE)[0]}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.show()

        print(f"\nSuccessfully processed {len(annotations)} parking polygons")
        print(f"Empty spaces: {empty_count}")
        print(f"Occupied spaces: {occupied_count}")
        print(f"Visualization saved to: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        if 'img' in locals():
            plt.imshow(img)
            plt.show()

def main():
    # Build paths
    image_path = os.path.join(DATASET_PATH, "train", "images", SAMPLE_IMAGE)
    label_path = os.path.join(DATASET_PATH, "train", "labels", 
                            SAMPLE_IMAGE.replace(".jpg", ".txt"))

    print(f"\nProcessing parking space visualization for:")
    print(f"Image: {image_path}")
    print(f"Labels: {label_path}")

    visualize_parking_polygons(image_path, label_path)

if __name__ == "__main__":
    # Check dependencies
    try:
        import cv2
        import matplotlib
    except ImportError:
        print("Required packages missing. Please install with:")
        print("pip install opencv-python matplotlib numpy")
        exit(1)

    main()