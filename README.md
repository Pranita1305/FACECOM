project:
  title: "Gender Classification using EfficientNet (TASK-A)"
  description: "Binary gender classification (female/male) using EfficientNet-B0, Albumentations, and PyTorch."

dataset:
  structure:
    root: "Comys_Hackathon5"
    train_dir: "Comys_Hackathon5/train"
    val_dir: "Comys_Hackathon5/val"
    classes: ["female", "male"]
  class_distribution:
    female: 79
    male: 343
    total: 422

libraries:
  - torch
  - torchvision
  - albumentations
  - albumentations.pytorch
  - efficientnet_pytorch
  - sklearn
  - matplotlib
  - seaborn
  - PIL

metrics:
  accuracy: 0.91
  precision:
    female: 0.72
    male: 0.95
    macro_avg: 0.84
    weighted_avg: 0.91
  recall:
    female: 0.81
    male: 0.93
    macro_avg: 0.87
    weighted_avg: 0.91
  f1_score:
    female: 0.76
    male: 0.94
    macro_avg: 0.85
    weighted_avg: 0.91

model:
  architecture: "EfficientNet-B0"
  pretrained: true
  final_layer:
    dropout: 0.4
    linear: "in_features → 2 (binary)"
  optimizer:
    name: "Adam"
    learning_rate: 1e-4
    weight_decay: 1e-4
  scheduler:
    type: "StepLR"
    step_size: 5
    gamma: 0.5
  loss_function:
    name: "CrossEntropyLoss"
    weighted: true

training:
  epochs: 5
  sampling: "WeightedRandomSampler"
  batch_size:
    train: 32
    val: 16
  device: "cuda if available, else cpu"
  checkpointing:
    best_model_metric: "val_f1"
    save_path: "best_model.pth"

augmentation:
  library: "albumentations"
  transforms:
    - Resize: [224, 224]
    - HorizontalFlip: 0.5
    - RandomBrightnessContrast: 0.5
    - CLAHE: 0.3
    - RandomShadow: 0.3
    - Normalize: "[mean, std]"
    - ToTensorV2

inference:
  example_image: "train/male/Abraham_Foxman_0001.jpg"
  function: "predict_image(image_path, model, transform, class_names)"
  output: "Predicted Gender"

outputs:
  best_model_weights: "best_model.pth"
  full_model: "gender_classification_model.pt"
  confusion_matrix: "generated in matplotlib"

project:
  title: "Face Identity Classification with MobileNetV2 (TASK-B)"
  description: >
    Robust facial identity recognition under varying visual distortions using MobileNetV2,
    CLAHE preprocessing, and Test-Time Augmentation (TTA).

dataset:
  base_path: "/content/drive/MyDrive/Comys_Hackathon5/Comys_Hackathon5/Task_B/train"
  image_size: [160, 160]
  batch_size: 32
  validation_split: 0.15
  class_mode: "categorical"
  augmentation:
    rotation_range: 20
    zoom_range: 0.2
    width_shift_range: 0.2
    height_shift_range: 0.2
    brightness_range: [0.5, 1.5]
    horizontal_flip: true
    fill_mode: "nearest"
  preprocessing:
    rescale: 1.0/255
    enhancement: "CLAHE (LAB color space)"

model:
  architecture: "MobileNetV2"
  input_shape: [160, 160, 3]
  pretrained: "ImageNet"
  frozen_layers: "All (initial training), Last 30 (fine-tuning)"
  final_layers:
    - "GlobalAveragePooling2D"
    - "Dropout(0.5)"
    - "Dense(NUM_CLASSES, activation='softmax')"
  optimizer:
    - phase: "initial"
      type: "Adam"
      lr: 0.001
    - phase: "fine-tuning"
      type: "Adam"
      lr: 0.00001
  loss_function: "categorical_crossentropy"
  metrics: ["accuracy"]
  callbacks:
    - "ModelCheckpoint(initial_model.h5)"
    - "EarlyStopping(val_loss, patience=3)"

training:
  epochs:
    initial: 5
    fine_tuning: 5
  validation_strategy: "flow_from_directory with validation_split"
  tta: true
  clahe: true
  enhancement_examples: 4
  confidence_threshold: 0.5

evaluation:
  overall:
    accuracy: 0.5 
    macro_f1_score: 0.5
  metrics:
    - classification_report
    - confusion_matrix
    - top1_accuracy
    - macro_f1_score
  per_condition_performance:
    blur:
      accuracy: 0.20 - 0.40
      macro_f1_score: 0.20 - 0.40
    fog:
      accuracy: 0.30 - 0.50
      macro_f1_score: 0.25 - 0.45
    rain:
      accuracy: 0.60 - 0.70
      macro_f1_score: 0.60 - 0.70
    low:
      accuracy: 0.55 - 0.65
      macro_f1_score: 0.55 - 0.65
    over:
      accuracy: 0.65 - 0.75
      macro_f1_score: 0.65 - 0.75
    sun:
      accuracy: 0.70 - 0.85
      macro_f1_score: 0.70 - 0.85
    glare:
      accuracy: 0.30 - 0.50
      macro_f1_score: 0.30 - 0.50
    light:
      accuracy: 0.40 - 0.60
      macro_f1_score: 0.40 - 0.60
    normal:
      accuracy: 0.80 - 0.90
      macro_f1_score: 0.80 - 0.90

inference:
  function: "predict_face_identity(image_path)"
  preprocessing: "CLAHE + TTA (original, flipped, brightness adjusted)"
  output: 
    - predicted_label
    - confidence_score
  visual_output: "Matplotlib plot with label and confidence"

outputs:
  saved_model: "initial_model.h5"
  metrics_plots:
    - "Training vs Validation Accuracy"
    - "Training vs Validation Loss"
    - "Confusion Matrix (Seaborn heatmap)"

utils:
  visualization:
    tool: "matplotlib"
    plot_types:
      - "confusion_matrix"
      - "training_curve"
  filenames_used: "val_gen.filenames"
  condition_extraction: "from filename substrings"

author: "Your Name Here"
license: "MIT"

