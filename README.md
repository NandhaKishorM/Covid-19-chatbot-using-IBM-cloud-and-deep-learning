# CHATBOT-Cough detection and automated response

## build librnnoise & rnnoise_demo with CMake

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

copy the file "rnnoise_demo" from "/build/bin/" to the main directory

download the weight file from https://drive.google.com/file/d/1BV2OSIuuwg6hx-22Q1ApayeWlf45AhmO/view?usp=sharing
and move it the "models" directory
## Edit the IBM credentials

1. Find the Assistant service in your IBM Cloud Dashboard Services.
2. Click on your Watson Assistant service and then click on Launch Watson Assistant.
3. Use the left sidebar and click on Assistants. Create an assistant.
4. Select the Dialog skill card and click Next.
5. Select the Import skill tab.
6. Click the Choose JSON File button and choose the data/covid_ai_chatbot_skill.json file in your cloned repo.
7. Click import
8. Go back to the Skills page (use the left sidebar).
9. Look for the created skill.
10. Click on the three dots in the upper right-hand corner of the card and select View API Details.
Copy the skill ID(workspace id). Go to your assitant click on the three dots in the upper right-hand corner and select settings and open API details note down the Assistant id( for android application ), assistant URL Save it for the next step.
11. Go to IBM functions and create an action with python 3.7 
12. Open function_action.py and copy-paste it in the Code section. Click on Parameters and add Parameter with Parameter name as "link" and Parameter Value as " https://api.covid19india.org/state_district_wise.json ".
13. ```
$ cp sample.env .env
```


## install anaconda python
```
$ conda create -n sound pip python=3.6
$ conda activate sound
$ pip install -r requirements.txt
```

## Run

```
$ python detail_live.py
```
# CHATBOT - Flask

```
$ python app.py
```
and go to 127.0.0.1:1880 from your browser

# CHATBOT-Node-Red

follow this tutorial https://developer.ibm.com/tutorials/create-a-voice-enabled-covid-19-chatbot-using-node-red/
and import flows.json from folder "node-red app"

# CHATBOT-Android application

Install app-debug.apk

# Emotion recognition
```
$ cd Emotion recognition
$ python main_script.py
```
## Principle
1. Dataset, FER2013 from Kaggle

2. Construct CNN with Keras using TensorFlow backend

3. Train the model from the given dataset

4. Face detection using Caffe based pre-trained deep learning model. Refer https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

5. Real-time emotion recognition and plot animated matplotlib graph from the output.

# Heart rate detection

```
$ cd Heart rate measurement
$ python GUI.py
```
Author : https://github.com/habom2310/Heart-rate-measurement-using-camera

## Principle

1. Face detection using dlib library and get the Region Of Interest(ROI)

2. Apply a band pass filter to eliminate a range of frequencies

3. Average colour value of the ROI calculated and pushed to a data buffer

4. Apply Fast Fourier Transform to the data buffer. Highest peak is the heart-rate

## Implementation in Jetson Nano


