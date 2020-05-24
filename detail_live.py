from keras.models import load_model
import tensorflow as tf
import numpy as np
from vggish_input import waveform_to_examples
import acoustics
import pyaudio
from pathlib import Path
import time
import argparse
import wget
import os
import wave
from reprint import output
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import subprocess
import pyttsx3
import shlex
from helpers import Interpolator, ratio_to_db, dbFS, rangemap

from ibm_watson import SpeechToTextV1
import os
from dotenv import load_dotenv
from ibm_watson import AssistantV1
from ibm_cloud_sdk_core import get_authenticator_from_environment
import assistant_setup
# One time initialization
engine = pyttsx3.init()

# Set properties _before_ you add things to say
engine.setProperty('rate', 150)    # Speed percent (can go over 100)
engine.setProperty('volume', 0.5)  # Volume 0-1

load_dotenv()


authenticator = (get_authenticator_from_environment('assistant') or
                 get_authenticator_from_environment('conversation'))
assistant = AssistantV1(version="2019-11-06", authenticator=authenticator)
workspace_id = assistant_setup.init_skill(assistant)



speech_to_text = SpeechToTextV1()
language = 'en'
def play_mp3(path):
    subprocess.Popen(['mpg123', '-q', path]).wait()
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "record.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(int(RATE / CHUNK * RECORD_SECONDS))

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    subprocess.call(shlex.split('./rnnoise_demo record.wav output.wav'))
    audio_file = open("output.wav", "rb")
  

    response = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            timestamps=True,
            word_confidence=True,
            smart_formatting=True).get_result()
 
    text_output = response['results']
    if len(text_output) == 0:
        print("no voice found")
    else:
       
        text_output = response['results'][0]['alternatives'][0]['transcript']
        text_output = text_output.strip()
        
        print(text_output)
        response = assistant.message(workspace_id=workspace_id,
                               input={'text':text_output}
                             )

        response = response.get_result()
        reponseText = response["output"]["text"]
        
        str1 = ''.join( reponseText)
        if len(str1)== 0 :

            engine.say("I didn't get that")


            engine.runAndWait()
           
        else:
            myobj = gTTS(text= str1 , lang=language, slow=False)
         
            myobj.save("result.mp3") 
            play_mp3("result.mp3")
      

 
# thresholds
PREDICTION_THRES = 0.85 # confidence
DBLEVEL_THRES = -30 # dB

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0
OUTPUT_LINES = 33

###########################
# Model download
###########################
def download_model(url,output):
    return wget.download(url,output)

###########################
# Check Microphone
###########################
print("=====")
print("1 / 2: Checking Microphones... ")
print("=====")

import microphones
desc, mics, indices = microphones.list_microphones()
if (len(mics) == 0):
    print("Error: No microphone found.")
    exit()

#############
# Read Command Line Args
#############
MICROPHONE_INDEX = indices[0]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mic", help="Select which microphone / input device to use")
args = parser.parse_args()
try:
    if args.mic:
        MICROPHONE_INDEX = int(args.mic)
        print("User selected mic: %d" % MICROPHONE_INDEX)
    else:
        mic_in = input("Select microphone [%d]: " % MICROPHONE_INDEX).strip()
        if (mic_in!=''):
            MICROPHONE_INDEX = int(mic_in)
except:
    print("Invalid microphone")
    exit()

# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    i = indices[k]
    if (i==MICROPHONE_INDEX):
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("2 / 2: Checking model... ")
print("=====")
model_filename = "models/example_model.hdf5"
acoustics_model = Path(model_filename)
if (not acoustics_model.is_file()):
    print("Downloading example_model.hdf5 [867MB]: ")
    download_model(MODEL_URL, MODEL_PATH)

##############################
# Load Deep Learning Model
##############################
print("Using deep learning model: %s" % (model_filename))
model = load_model(model_filename)
graph = tf.get_default_graph()
context = acoustics.everything

label = dict()
for k in range(len(context)):
    label[k] = context[k]

##############################
# Setup Audio Callback
##############################
output_lines = []*OUTPUT_LINES
audio_rms = 0
candidate = ("-",0.0)

# Prediction Interpolators
interpolators = []
for k in range(31):
    interpolators.append(Interpolator())

# Audio Input Callback
def audio_samples(in_data, frame_count, time_info, status_flags):
    global graph
    global output_lines
    global interpolators
    global audio_rms
    global candidate
    np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0 # Convert to [-1.0, +1.0]

    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)
    interp = interpolators[30]
    interp.animate(interp.end, db, 1.0)

    # Make Predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    with graph.as_default():
        if x.shape[0] != 0:
            x = x.reshape(len(x), 96, 64, 1)
            pred = model.predict(x)
            predictions.append(pred)

        for prediction in predictions:
            m = np.argmax(prediction[0])
            candidate = (acoustics.to_human_labels[label[m]],prediction[0,m])
            num_classes = len(prediction[0])
            for k in range(num_classes):
                interp = interpolators[k]
                prev = interp.end
                interp.animate(prev,prediction[0,k],1.0)
    return (in_data, pyaudio.paContinue)

##############################
# Main Execution
##############################
while(1):
    ##############################
    # Setup Audio
    ##############################
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_samples, input_device_index=MICROPHONE_INDEX)

    ##############################
    # Start Non-Blocking Stream
    ##############################
    os.system('cls' if os.name == 'nt' else 'clear')
    print("# Live Prediction Using Microphone: %s" % (mic_desc))
    stream.start_stream()
    while stream.is_active():
        with output(initial_len=OUTPUT_LINES, interval=0) as output_lines:
            while True:
                time.sleep(1.0/FPS) # 60fps
                for k in range(30):
                    interp = interpolators[k]
                    val = interp.update()
                    bar = ["|"] * int((val*100.0))
                    output_lines[k] = "%20s: %.2f %s" % (acoustics.to_human_labels[label[k]], val, "".join(bar))

                # dB Levels
                interp = interpolators[30]
                db = interp.update()
                val = rangemap(db, -50, 0, 0, 100)
                bar = ["|"] * min(100,int((val)))
                output_lines[30] = "%20s: %.1fdB [%s " % ("Audio Level", db, "".join(bar))

                # Display Thresholds
                output_lines[31] = "%20s: confidence = %.2f, db_level = %.1f" % ("Thresholds", PREDICTION_THRES, DBLEVEL_THRES)

                # Final Prediction
                pred = "-"
                event,conf = candidate
                if (conf > PREDICTION_THRES and db > DBLEVEL_THRES):
                    pred = event
                output_lines[32] = "%20s: %s" % ("Prediction", pred.upper())
                if pred=="Person Talking":
                    
                    play_mp3("beep.mp3")
                    record_audio()
                elif pred == "Coughing":
                    myobj = gTTS(text= "Sir, you are coughing" , lang=language, slow=False)
         
                    myobj.save("cough_detect.mp3") 
                    play_mp3("cough_detect.mp3")
