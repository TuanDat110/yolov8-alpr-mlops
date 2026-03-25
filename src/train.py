import mlflow
from ultralytics import YOLO
import os

def main():
    # 1. Khởi tạo cấu hình MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("YOLOv8_License_Plate_Detection")
    
    # 2. Định nghĩa Hyperparameters (ĐÃ HẠ NHIỆT ĐỂ CỨU RAM)
    params = {
        "model_name": "yolov8n.pt", 
        "data_path": "data/data.yaml",
        "epochs": 50,
        "batch_size": 8,     # <--- Giảm xuống 8 để không bị tràn RAM (OOM Killed)
        "imgsz": 640,
        "device": "0", 
        "workers": 2         # <--- Lưu ở đây là đúng rồi
    }

    # 3. Đổi tên Run để MLflow không bị nhầm lẫn
    with mlflow.start_run(run_name="YOLOv8_V2"):
        mlflow.log_params(params)
        
        try:
            print("Initialize the YOLOv8 model...")
            model = YOLO(params["model_name"])
            
            print("Begin the training process...")
            results = model.train(
                data=params["data_path"],
                epochs=params["epochs"],
                batch=params["batch_size"],
                imgsz=params["imgsz"],
                device=params["device"],
                workers=params["workers"],  # <--- BẮT BUỘC PHẢI THÊM DÒNG NÀY NÓ MỚI HIỂU!
                project="runs/train",
                name="alpr_exp_v2"          # <--- ĐỔI TÊN ĐỂ MLFLOW KHÔNG BÁO LỖI "BAD REQUEST"
                # Đã xóa exist_ok=True để nó lưu vào folder mới
            )
            
            # 5. Log Metrics
            mlflow.log_metric("mAP50", results.box.map50)
            mlflow.log_metric("mAP50-95", results.box.map)
            
            # 6. Log Artifacts (Nhớ trỏ đúng về folder mới)
            best_model_path = "runs/train/alpr_exp_v2/weights/best.pt"
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path, artifact_path="models")
                print(" Model best.pt has been uploaded to MLflow Server!")
            else:
                print(" No model file found to upload to MLflow.")
                
        except Exception as e:
            print(f"Errors during training: {e}")
            mlflow.log_param("status", "failed")

if __name__ == "__main__":
    main()