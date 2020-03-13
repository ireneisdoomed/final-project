
import cv2
import numpy as np
import time
from gestureDetection import detectGesture

def captureVideo(camera=0):
    '''
    Captures and saves video from a computer camera.
    input: computer camera is default, any connected camera can be selected by index
    '''
    # Sets up a connection with the computer camera
    capture = cv2.VideoCapture(camera)

    # Width and height of the frame
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Defines the codec and creates VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'XVID')
    recorder = cv2.VideoWriter('../output/videos/output.avi',codec, 25.0, (width,height))

    while True:
        # Constantly reads the capture and stores it in the video variable
        hasFrames, video = capture.read()

        # Saves the video
        recorder.write(video)

        # Displays the resulting frame until 'q' is pressed
        cv2.imshow("Video", video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture object is deleted and windows are closed
    capture.release()
    recorder.release()
    cv2.destroyAllWindows()
    print('Recording saved with an output.avi filename.')

def captureAndPredict(camera = 0):
    '''
    Captures and saves video from a computer camera.
    input: computer camera is default, any connected camera can be selected by index
    '''
    while True:
        #ini_time = time.clock()
        # Sets up a connection with the computer camera
        capture = cv2.VideoCapture(camera)

        # Width and height of the frame
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        hasFrames, video = capture.read()
        if hasFrames:
            #cv2.imwrite("../data/ruido_fotos/pic{}.jpg".format(str(count)), video)
            detectGesture('', video)
            capture.release()
        #fin_time = time.clock()
        #duration = fin_time- ini_time
        #time.sleep(10-duration/1000)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

