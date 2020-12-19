# Facial Expression recognition

Data is from Kaggle:
```
https://www.kaggle.com/c/emotion-detection-from-facial-expressions/data
```
Dlib pretrained model:
```
http://dlib.net/face_landmark_detection.py.html
```
How to use this project:
```
0. All the code should be downloaded and run from Google Drive link: https://drive.google.com/drive/u/0/folders/1Q4sR71mDFOjpKNtmFBXjDYzfZzHJ__D3
1. The code here are just backups, the code are in respective positions in the google drive.
2. You need to have a python interpreter with cv2, numpy, plotly, dlib, sklearn, imutils packages installed.
3. Open the Facial-Expression-Recognition folder in a python IDE, i.e. Pycharm.
	Run the KNN.py to generate the results for KNN showed in the report;
	Run the SVM.py to generate the results for SVM showed in the report;
	Run the Load_Data.py to generate the Default face images showed in the report.
   Note that to test differnet methods, please comment and uncomment lines in the main function.
   Where input 0 uses Gradient, 1 uses Edge, 2 uses HoG and 3 uses Facial Landmarks.
4. Run the vgg_main.ipynb on the jupyter notebook to see step by step results.
5. The report have a detailed explanation of the project.