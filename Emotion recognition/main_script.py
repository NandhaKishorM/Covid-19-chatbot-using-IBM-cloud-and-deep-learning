import numpy as np
import cv2
from keras.preprocessing import image
import time
import multiprocessing
import csv
from tempfile import NamedTemporaryFile
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from imutils.video import VideoStream
import argparse
import imutils
from PIL import Image
import io



def main():
        
        
              #-----------------------------
        #opencv initialization

        

        #-----------------------------
        #face expression recognizer initialization
        from keras.models import model_from_json
        model = model_from_json(open("facial_expression_model_structure.json", "r").read())
        model.load_weights('facial_expression_model_weights.h5') #load weights
        #-----------------------------

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--prototxt", default="deploy.prototxt.txt",
                help="path to Caffe 'deploy' prototxt file")
        ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
                help="path to Caffe pre-trained model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

        # initialize the video stream and allow the cammera sensor to warmup
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)
       
        
        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                (ret,frame) = vs.read()
                frame = cv2.resize(frame, (600,600))
                img = frame
         
                # grab the frame dimensions and convert it to a blob
                (h, w) = frame.shape[:2]
                print(w)
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                        (300, 300), (104.0, 177.0, 123.0))
         
                # pass the blob through the network and obtain the detections and
                # predictions
                net.setInput(blob)
                detections = net.forward()

                # loop over the detections
                for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with the
                        # prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the `confidence` is
                        # greater than the minimum confidence
                        if confidence < args["confidence"]:
                                continue

                        # compute the (x, y)-coordinates of the bounding box for the
                        # object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                       
                    
                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{:.2f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        #cv2.rectangle(frame, (startX, startY), (endX, endY),
                               #(0, 0, 255), 2)
                        #cv2.putText(frame, text, (startX, y),
                              # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        #cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face
                
                                
                        detected_face = img[int(startY):int(endY), int(startX):int(endX)] #crop detected face
                        if detected_face.size > 0:
                                
                                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                                detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
                                
                                img_pixels = image.img_to_array(detected_face)
                                img_pixels = np.expand_dims(img_pixels, axis = 0)
                                
                                img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                                
                                #------------------------------
                                
                                predictions = model.predict(img_pixels) #store probabilities of 7 expressions
                                
                                
                                max_index = np.argmax(predictions[0])
                                
                                #background of expression list
                                overlay = img.copy()
                                opacity = 0.4
                                cv2.rectangle(img,(endX+10,startY-25),(endX+150,startY+115),(64,64,64),cv2.FILLED)
                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                                
                                #connect face and expressions
                                cv2.line(img,(int((startX+endX)/2),startY+15),(endX,startY-20),(255,255,255),1)
                                cv2.line(img,(endX,startY-20),(endX+10,startY-20),(255,255,255),1)
                                
                                emotion = ""
                                for i in range(len(predictions[0])):
                                        emotion = "%s %s%s" % (emotions[i], round(predictions[0][i]*100, 2), '%')
                                        '''print(i)
                                        print(emotion)
                                        a = emotions[i]
                                        b = int(round(predictions[0][i]*100, 0))
                                        
                                        with open('detection.csv', 'a', newline='') as file:
                                                writer = csv.writer(file)
                                                writer.writerow([a, b])'''
                                        filename = 'detection.csv'
                                        tempfile = NamedTemporaryFile(mode='w', delete=False)
                                        stud_ID=emotions[i]
                                        stud_year = int(round(predictions[0][i]*100, 0))
                                        if stud_year > int(60) or stud_year < int(10):
                                                
                                                

                                                fields = ['null','0']

                                                with open(filename, 'r') as csvfile, tempfile:
                                                    reader = csv.DictReader(csvfile, fieldnames=fields)
                                                    writer = csv.DictWriter(tempfile, fieldnames=fields)
                                                    for row in reader:
                                                        if row['null'] == str(stud_ID):
                                                            print('updating row', row['null'])
                                                            row['0']= stud_year
                                                        row = {'null': row['null'], '0': row['0']}
                                                        writer.writerow(row)

                                                shutil.move(tempfile.name, filename)

                                        """if i != max_index:
                                                color = (255,0,0)"""
                                                
                                        color = (255,255,255)
                                        
                                        cv2.putText(img, emotion, (int(endX+15), int(startY-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                
                                
                        #-------------------------
          
          
                cv2.imshow('img',img)

                frame = frame + 1
                #print(frame)

                #---------------------------------



                if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                        break

        #kill open cv things
        vs.release()
        cv2.destroyAllWindows()
def graph():
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    emotions =  ['null','angry','disgust','fear', 'happy','sad','surprise','neutral' ]
    x_pos = np.arange(len(emotions))

    def animate(i):
        pullData = open("detection.csv","r").read()
        dataArray = pullData.split('\n')
        xar = []
        yar = []
        for eachLine in dataArray:
            if len(eachLine)>1:
                
                x,y = eachLine.split(',')
                xar.append(str(x))
                yar.append(int(y))
        ax1.clear()
   
        ax1.bar(x_pos,yar,width=0.5,color="red")
    
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(emotions)
    
        plt.title('Emotion analysis')
        plt.ylabel('Percentage')
        plt.xlabel('Emotion')
    ani = animation.FuncAnimation(fig, animate, interval=5)
    plt.show()
if __name__ == "__main__": 
    # creating processes 
    p1 = multiprocessing.Process(target=main) 
    p2 = multiprocessing.Process(target=graph) 
  
    # starting process 1 
    p1.start() 
    # starting process 2 
    p2.start() 
  
    # wait until process 1 is finished 
    
    # wait until process 1 is finished 
    p1.join() 
    # wait until process 2 is finished 
    p2.join() 
  
    # both processes finished 
    print("Done!") 
    
