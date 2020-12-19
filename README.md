# Facial Expression recognition

## Start with data

Extrat data from the og_data file.

Data is from Kaggle:
```
https://www.kaggle.com/c/emotion-detection-from-facial-expressions/data
```
Haarcascades Model is from OpenCV:  # Note that it is not used in the project. We used dlib instead.
```
https://github.com/AlexeyAB/OpenCV-detection-models
```
Dlib pretrained model:
```
http://dlib.net/face_landmark_detection.py.html
```
How to use this project:
```
1. You need to have a python interpreter with cv2, numpy, plotly, dlib, sklearn, imutils packages installed.
2. Open the kNN_and_SVM folder in a python IDE, i.e. Pycharm.
	Run the KNN.py to generate the results for KNN showed in the report;
	Run the SVM.py to generate the results for SVM showed in the report;
	Run the Load_Data.py to generate the Default face images showed in the report.
   Note that to test differnet methods, please comment and uncomment lines in the main function.
   Where input 0 uses Gradient, 1 uses Edge, 2 uses HoG and 3 uses Facial Landmarks.
3. The report have a detailed explanation of the project.