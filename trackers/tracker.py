from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.cls_names = self.model.names  # Store class names when initializing the model

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            break  # You might want to remove this `break` if you want to process all frames
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        
        # Loop through each detection in the batch
        for frame_num, detection in enumerate(detections):
            # Access class names via the stored class names from the model
            cls_names_inv = {v: k for k, v in self.cls_names.items()}
            print(cls_names_inv)

            # Create supervision detection from YOLO detection
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Goalkeeper to player id 
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                # Use the correct way to access the class name using the class_id
                if self.cls_names[class_id] == 'goalkeeper':
                    # Change goalkeeper to player
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Debugging output: Print detection details
            print(f"Frame {frame_num}: {detection_supervision}")
            break
