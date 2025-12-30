from ultralytics import YOLO
import os

def extract_yolo_classes():
    try:
        # Load YOLOv11 model
        print("Loading YOLOv11 model...")
        model = YOLO("models/yolo11x.pt")
        
        # Get class names from the model
        class_names = model.names
        
        # Sort by index to ensure correct order
        sorted_classes = [class_names[i] for i in range(len(class_names))]
        
        # Format 1: Original array format
        classes_str = "YOLO_CLASSES = ["
        for i, name in enumerate(sorted_classes):
            if i % 5 == 0:  # Start new line every 5 items
                classes_str += "\n                "
            classes_str += f"'{name}', "
        classes_str = classes_str.rstrip(", ")  # Remove trailing comma and space
        classes_str += "]"
        
        # Write array format to file
        with open('yolo_classes.txt', 'w') as f:
            f.write(classes_str)
            
        # Format 2: Class ID and Name format
        with open('yolo_classes_with_ids.txt', 'w') as f:
            for i, name in enumerate(sorted_classes):
                f.write(f"Class ID: {i}, Class Name: {name}\n")
        
        print(f"\nClasses have been written to:")
        print("1. 'yolo_classes.txt' (array format)")
        print("2. 'yolo_classes_with_ids.txt' (ID and name format)")
        print(f"Total number of classes: {len(sorted_classes)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Disable auto-download of models
    os.environ['YOLO_AUTOINSTALL'] = '0'
    
    extract_yolo_classes()