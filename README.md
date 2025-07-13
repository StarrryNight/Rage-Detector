# RAGE Detector 
A real-time emotion detection system designed to identify rage and anger in gaming contexts, particularly for detecting bad randoms in Marvel Rivals and other competitive games.

This project uses computer vision and machine learning to analyze facial expressions, body posture, and hand gestures in real-time to detect when a player is experiencing rage or anger. The system leverages MediaPipe for pose and facial landmark detection, combined with trained machine learning models to classify emotional states.



## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Rage-Detector
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


### Running the Rage Detector

1. **Start the main application**
   ```bash
   cd main
   python main.py
   ```

2. **Use the application**
   - The webcam feed will open with real-time landmark detection
   - Emotion classification results will be displayed on screen
   - Press 'q' to quit the application

### Training Your Own Model

1. **Collect training data**
   ```bash
   cd training
   python collectData.py
   ```
   - Modify the `current_feeling` variable in `collectData.py` to collect different emotions
   - Set `csv_initialized = False` to start fresh data collection
   - The system will capture pose and facial landmarks while you express different emotions
   - ***Remember to set `csv_initialized = True` after initializing the file once

2. **Train the model**
   ```bash
   python train.py
   ```
   - This will train multiple machine learning models on your collected data
   - Models are saved as pickle files for later use

3. **Handle CSV data (optional)**
   ```bash
   python handleCSV.py
   ```
   - Use this script to process and clean your training data

## Project Structure

```
Rage-Detector/
├── main/                   # Main application files
│   ├── main.py            # Real-time detection application
│   └── getResult.py       # Result processing utilities
├── training/              # Model training and data collection
│   ├── collectData.py     # Data collection script
│   ├── train.py          # Model training script
│   ├── handleCSV.py      # CSV data processing
│   ├── data.csv          # Training dataset
│   ├── LR.pkl            # Logistic Regression model
│   ├── RF.pkl            # Random Forest model
│   └── GB.pkl            # Gradient Boosting model
├── requirements.txt       # Python dependencies
├── LICENSE               # Project license
└── README.md            # This file
```


### Model Selection
The system currently uses Logistic Regression by default. Use other models by:

1. Uncomment the desired model in `training/train.py`
2. Retrain the model
3. Update the model loading in `main/main.py`

**HAPPY RAGING BECAUSE OF BAD RANDOMS!!!**
