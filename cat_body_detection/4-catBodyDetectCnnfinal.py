from ultralytics import YOLO

def main():
    # Load YOLO Pretrained Model
    yolo_model = YOLO('yolov8n.pt')  # Use YOLOv8 nano for lightweight and efficient training

    # Train YOLO Model
    yolo_model.train(
        data='./cat_face_detection/yaml.yaml',  # Path to the dataset configuration file
        epochs=30,
        imgsz=640,  # Resize images to 640x640 for training
        batch=16,  # Batch size
        workers=4,  # Number of data loading workers
        save_dir="./cat_face_detection/output",  # Directory to save model and logs
        project='cat_face_detection',
        name='cat_detector'
    )

    # Save the Model
    yolo_model.export(format="saved_model", dynamic=False)


if __name__ == "__main__":
    main()
