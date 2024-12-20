import cv2
import time
import mediapipe as mp
from scipy.spatial import distance as dist
from model import FacialExpressionModel
import numpy as np
import VJ
from VideoGet import VideoGet
import threading
import queue
from playsound import playsound
import visualizer


#load models
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")





# Time delays after each smile detection (in seconds)
delays = [45,38,44,54, 34, 37 ,63, 55, 13, 36, 36, 36]


# Counters for different detections
global smile_sizes, consecutive_angry, consecutive_suprises, consecutive_happy, smileQuality, fakeness, confirmation, switching_event
suprises_count = 0
connfirmation = False
switching_event = threading.Event()

consecutive_smiles = 0
consecutive_suprises = 0
consecutive_happy = 0
consecutive_angry = 0
smile_sizes = [0 , 0 , 0]

smileQuality = 0
fakeness =0


#Bool for the time to check for suprise instead of smile
checkForSuprise = False
section = 0

# Constants for suprise detection
THRESHOLD = 0.2  # Threshold for detecting an open mouth
CONSECUTIVE_FRAMES = 5  # Number of consecutive frames for which the mouth must be open to trigger the message
EYE_BLINK_HEIGHT = .15
EYE_SQUINT_HEIGHT = .18
EYE_OPEN_HEIGHT = .25
EYE_BUGGED_HEIGHT = .7

MOUTH_OPEN_HEIGHT = .2
MOUTH_OPEN_SHORT_FRAMES = 1
MOUTH_OPEN_LONG_FRAMES = 4
MOUTH_CLOSED_FRAMES = 1

MOUTH_FROWN = .006
MOUTH_NOSE_SCRUNCH = .09
MOUTH_SNARL = .1
MOUTH_DUCKFACE = 1.6

BROW_RAISE_LEFT = .0014
BROW_RAISE_RIGHT = .014
BROWS_RAISE = .16


EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


duckfacing = False

brows_raised = False
brows_raised_count = 0
brows_raised_frames = 0

pred = ''
frame_queue = queue.Queue()


def get_aspect_ratio(top, bottom, right, left):
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width



def detectHappy(consecutive_happy, section):

    # Increase consecutive smile counter if smiling
    if 'Happy' in pred and current_pred['Happy'] > 0.57:
        consecutive_happy += 1
        print(consecutive_happy)
    else:

        consecutive_happy = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_happy == 7:
        print(f"10 consecutive happy frames detected. Sending MIDI note")
        section +=1
        #section = 9
        consecutive_happy = 0  # reset the consecutive smile counter
        switching_event.set()
        VJ.start_siwtch()
        VJ.selectVideo(section, 0)
        print('will play: '+ str(section))
       
        
        #playsound('confirm.wav')
        
        return section, consecutive_happy, time.time()
    return section, consecutive_happy, None


#Surprised face detection using emotion detection model and openCV
def detectSuprised(consecutive_suprises, section, pred_current):
    print(pred_current)
    # Increase consecutive smile counter if smiling
    if 'Surprise' in pred_current or current_pred['Surprise'] >0.25:
        consecutive_suprises += 1
    else:
        # Reset consecutive smile counter if not smiling
        consecutive_suprises = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_suprises == 5:
        print(f"10 consecutive suprised frames detected. Sending MIDI note")
        section +=1
        consecutive_suprises = 0  # reset the consecutive smile counter
        
        VJ.selectVideo(section, 0)
        #playsound('confirm.wav')
        return section, consecutive_suprises, time.time()



    return section, consecutive_suprises, None


    
def detectAngry(consecutive_angry, section):
    
    # Increase consecutive smile counter if smiling
    if 'Angry' in pred or current_pred['Angry'] >0.22:
        consecutive_angry += 1
    else:
        # Reset consecutive smile counter if not smiling
        consecutive_angry = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_angry == 8:
        print(f"10 consecutive angry frames detected. ")
        section +=1
        consecutive_angry = 0  # reset the consecutive smile counter
        
        VJ.selectVideo(section, 0)
       

        return section, consecutive_angry, time.time()
    return section, consecutive_angry, None
def mouthOpen(face, image):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh


    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
        
            for face_landmarks in results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                face = face_landmarks.landmark

                # Get the coordinates of the upper and lower lip landmarks
                upper_lip_landmarks = [face_landmarks.landmark[i] for i in range(13, 16)]
                lower_lip_landmarks = [face_landmarks.landmark[i] for i in range(14, 17)]

                # Calculate the average vertical position of the upper and lower lip landmarks
                upper_lip_avg_y = sum([landmark.y for landmark in upper_lip_landmarks]) / len(upper_lip_landmarks)
                lower_lip_avg_y = sum([landmark.y for landmark in lower_lip_landmarks]) / len(lower_lip_landmarks)
                mouth_distance = lower_lip_avg_y - upper_lip_avg_y

                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                )

                mouth_inner_top = face[13]
                mouth_inner_bottom = face[14]
                mouth_inner_right = face[78]
                mouth_inner_left = face[308]
                mouth_inner_ar = get_aspect_ratio(
                    mouth_inner_top, mouth_inner_bottom, mouth_inner_right, mouth_inner_left)
                return mouth_inner_ar
    #return how wide open the mouth is
    return None
def evaluateSmileQuality(mouthGap, happyProb):
    #mouthGap is considered 'fake' if it's bigger than 0.32. Make a conversion of that value from 0 to 100,
    #such that if mouthGap is bigger than 0.32 the conversio should results in a value of 50
    oldMin = 0.1
    oldMax=0.8

    fakeness = ((mouthGap - oldMin) * 100) / (oldMax - oldMin)
   
    happinessQuality = happyProb*100 - (fakeness *0.4)

    #return ("%.2f" % fakeness), ("%.2f" % happinessQuality)
    return int(fakeness), int(happinessQuality)



def detectSmileSize(faces, grayscale, img, smile_sizes, section):
    global smileQuality, fakeness
    
    mouthGap = mouthOpen(faces, img)
   
    answers = ["Eeew, what an ugly smile. Tone it down babes.", "You can make that smile bigger, honey. Let your positive energy shine the world!", "You just got the perfect smile!"]
    #using emotion detection (happy) instead of smile detection
    if 'Happy' in pred:
    #if(detectSmile(faces,grayscale, img)):
        
        if(mouthGap > 0.32):
            print("Eeew, what an ugly smile. Tone it down babes.")
            smile_sizes[0] = smile_sizes[0] + 1
            smile_sizes[1] = 0
            smile_sizes[2] = 0

        elif(mouthGap < 0.2):
            print("You can make that smile bigger, honey. Let your positive energy shine the world!")
            smile_sizes[0] = 0
            smile_sizes[1] = smile_sizes[1] + 1
            smile_sizes[2] = 0
        else: 
            print("You just got the perfect smile!")
            smile_sizes[0] = 0
            smile_sizes[1] = 0
            smile_sizes[2] = smile_sizes[2] + 1
        
        print("Consecutive: " + str(max(smile_sizes)))
    else:
        print("You are not smiling and I see it.")
        smile_sizes = [0,0,0]

    if max(smile_sizes) == 3:
        
        fakeness, smileQuality = evaluateSmileQuality(mouthGap, current_pred['Happy'])

 
        print(answers[smile_sizes.index(max(smile_sizes))])
        section += (1 + smile_sizes.index(max(smile_sizes)))
        time.sleep(2)
        VJ.selectVideo(section, 2)
        smile_sizes = [0,0,0]
     
        return section, smile_sizes,  time.time()
    
    return section, smile_sizes,  None



def holdDetection():
    global confirmation
    return

global noSmile




last_time = None
end= False
noSmile = False
last_section = 0
last_change_time = time.time()
delays = [2,2,2, 2, 2, 2 ,2, 2, 9, 36, 36, 36, 36]
delays = [4,4,4, 4, 4, 4 ,4, 4, 9, 36, 36, 36, 36]

delays = [41,37,33, 52, 33, 34 ,63, 54, 12, 36, 36, 36, 36]


stop_event = threading.Event()
emotion_timer = time.time()
visualizer.init()
time.sleep(3)
end_time = 0
eval = False
def video_processing():
    global eval, end_time, switching_event, noSmile, smileQuality, fakeness, section, last_time, end, last_section, last_change_time, consecutive_smiles, pred, emotion_timer, happiness_predictions, current_pred, previous_pred, consecutive_suprises, consecutive_angry, consecutive_happy, smile_sizes
    
    video_getter = VideoGet(0).start()#
   
    
    while not stop_event.is_set():
        
        img = video_getter.frame
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_face.detectMultiScale(grayscale, 1.3, 5)
        
        if( faces == ()):
            scoreboard_img_cv = visualizer.noFace()
            cv2.imshow('Scoreboard', scoreboard_img_cv)
            pred = 'Sad'

        
        if(VJ.checkEnds()):
            
            if(section <= 9 and section != 0):
                section = 0
                VJ.selectVideo(section, 1)
            elif section > 9:
                noSmile = True
                cv2.destroyWindow("Smile Quality")
                section = 0
                VJ.selectVideo(section, 2)

            smileQuality = 0
            fakeness =0
            
            
            time.sleep(1)
            noSmile = False

        if VJ.check_switch():
            switching_event.clear()

        for (x, y, w, h) in faces:
            #check emotion only every second
            if time.time() - emotion_timer >= 1 and not switching_event.is_set():
                fc = grayscale[y:y+h, x:x+w]

                roi = cv2.resize(fc, (48, 48))
                
                # pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                pred_array = model.predict_emotion_array(roi[np.newaxis, :, :, np.newaxis])
                current_pred = {emotion: prob for emotion, prob in zip(EMOTIONS_LIST, pred_array[0])}

                pred = FacialExpressionModel.EMOTIONS_LIST[np.argmax(pred_array)]
                plot_img, scoreboard_img_cv = visualizer.main(pred_array)

                
                 #note:S happines chart only start if there has been a face detected
                cv2.imshow('Happiness Trend', plot_img)
                if faces != ():
                    cv2.imshow('Scoreboard', scoreboard_img_cv)
                emotion_timer = time.time()

        if(section  > 9):
            smileQuality_img = visualizer.showSmileQuality(section, fakeness, smileQuality)
            """if( not eval):
                end_time = time.time() + 15
                eval = True
            if not time.time() >= end_time:"""
            cv2.imshow('Smile Quality', smileQuality_img)
            



        if last_time is None or (time.time() - last_time > delays[section - 1]):
          
            if section == 0: #smile to play
               
                #section, consecutive_smiles, last_time = detectConsecutiveSmile(faces, grayscale, img, section, consecutive_smiles)
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
                
            elif section == 1: #smile to continue
               
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
                #section, smile_sizes, last_time = detectSmileSize(faces, grayscale, img, smile_sizes, section)
                
            elif section == 2: #smile to continue
                
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)

            elif section == 3: #be surprised
               
                #image_frame, consecutive_suprises, last_time = suprise_detector.detect_suprise(img, consecutive_suprises)
                section, consecutive_suprises, last_time = detectSuprised(consecutive_suprises, section, pred)

            elif section == 4: #laughter detection
                
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)

            elif section == 5: #smile to consent
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)

            elif section == 6: #be happy to agree
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
                
            elif section == 7: #smile if you understood
                
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)

            elif section == 8: #be angry
                section, consecutive_angry, last_time = detectAngry(consecutive_angry, section)
                #section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
            elif section == 9 and not noSmile: #last smile test
                end = True
                section, smile_sizes, last_time = detectSmileSize(faces, grayscale, img, smile_sizes, section)
                print(section)

            if last_time is not None and not end:
                time_remaining = delays[section - 1] - (time.time() - last_time)
                print(f"Waiting for next detection. Time remaining: {time_remaining:.2f} seconds")
                

            
        
        frame_queue.put(img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_getter.stop()
            stop_event.set()
           
            break

def display_frame():
    while not stop_event.is_set():
        if not frame_queue.empty():
            img = frame_queue.get()
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                print('stopped2')
                cv2.destroyAllWindows()
                break
thread_vj = threading.Thread(target=VJ.main)
thread_video = threading.Thread(target=video_processing)


# Starting the threads
thread_vj.start()

#time.sleep(2)
thread_video.start()


# Display frames in main thread
display_frame()

# Waiting for both threads to finish
thread_video.join()
thread_vj.join()

#root.mainloop()
