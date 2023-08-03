import cv2
import mido
import time
from mido import Message
import mediapipe as mp
from scipy.spatial import distance as dist
from datetime import datetime
from model import FacialExpressionModel
import numpy as np
import VJ
from VideoGet import VideoGet
import threading
import queue
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound

import matplotlib.pyplot as plt

import seaborn as sns
from PIL import Image, ImageFont, ImageDraw

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import keyboard

sns.set()

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
last_time = time.time()
previous_pred = {emotion: 0 for emotion in EMOTIONS_LIST}

fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
line, = ax.plot([], [], color='hotpink', linewidth=2)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.set_ylabel("Happiness")
canvas = FigureCanvas(fig)

happiness_predictions = []



cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font_emotion = cv2.FONT_HERSHEY_SIMPLEX
# Setting up MIDI
#port = mido.open_output('virtual port', virtual=True)

# Time delays after each smile detection (in seconds)
delays = [45,38,44,54, 34, 37 ,63, 55, 13, 36, 36, 36]

# MIDI notes to send for each smile


# Counter for smile detection
#smile_count = 0



# Counter for suprise detection
suprises_count = 0

global smile_sizes, consecutive_angry, consecutive_suprises, consecutive_happy


# Counter for consecutive smile detections
consecutive_smiles = 0
# Counter for consecutive surprise detections
consecutive_suprises = 0
consecutive_happy = 0
consecutive_angry = 0

smile_sizes = [0 , 0 , 0]

#Bool for the time to check for suprise instead of smile
checkForSuprise = False
section = 0

checkForSuprise = False
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
#Simle Detection only
def detectSmile(face, grayscale, img):
    is_smiling = False
  
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
      
        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile:
           
            cv2.rectangle(ri_grayscale, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)
            is_smiling = True
    return is_smiling

def detectConsecutiveSmile(face, grayscale, img, section, consecutive_smiles):
    #face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
   
    # Increase consecutive smile counter if smiling
    if detectSmile(face, grayscale, img):
        print("smiling")
        consecutive_smiles += 1
       # print(consecutive_smiles)
    else:
        # Reset consecutive smile counter if not smiling
        print("not smiling")
        consecutive_smiles = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_smiles == 10:
        print(f"15 consecutive smiles detected. Sending MIDI note")
        #msg = Message('note_on', note=midi_notes[section])
        #port.send(msg)
        #smile_count += 1
        section +=1 
        consecutive_smiles = 0  # reset the consecutive smile counter
        #VJ.play_pause()
        #VJ.affect()
        VJ.selectVideo(section)
        return section, consecutive_smiles, time.time()



    return section, consecutive_smiles, None


def detectRaisedBrows(face):
        eyeR_top = face[159]

        eyeL_top = face[386]

        browR_top = face[52]
        browR_bottom = face[223]
        browR_eyeR_lower_dist = dist.euclidean([browR_bottom.x, browR_bottom.y],
            [eyeR_top.x, eyeR_top.y])
        browR_eyeR_upper_dist = dist.euclidean([browR_top.x, browR_top.y],
            [eyeR_top.x, eyeR_top.y])
        browR_eyeR_dist = (browR_eyeR_lower_dist + browR_eyeR_upper_dist) / 2

        browL_top = face[443]
        browL_bottom = face[257]
        browL_eyeL_lower_dist = dist.euclidean([browL_bottom.x, browL_bottom.y],
            [eyeL_top.x, eyeL_top.y])
        browL_eyeL_upper_dist = dist.euclidean([browL_top.x, browL_top.y],
            [eyeL_top.x, eyeL_top.y])
        browL_eyeL_dist = (browL_eyeL_lower_dist + browL_eyeL_upper_dist) / 2
        brows_relative_raise = browR_eyeR_dist - browL_eyeL_dist

        if brows_relative_raise < BROW_RAISE_LEFT:
            return True
                
        if brows_relative_raise > BROW_RAISE_RIGHT:
            return True
        
        return False


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


#Surprised face detection using emotion detection model and openCV
def detectSuprised(consecutive_suprises, section, pred_current):
    print(pred_current)
    # Increase consecutive smile counter if smiling
    if 'Surprise' in pred_current:
        #print("happy")
        
        consecutive_suprises += 1
        #print(consecutive_suprises)
    else:
        # Reset consecutive smile counter if not smiling
        #print("not suprised")
        consecutive_suprises = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_suprises == 4:
        print(f"10 consecutive suprised frames detected. Sending MIDI note")

  
        section +=1
        consecutive_suprises = 0  # reset the consecutive smile counter
        
        VJ.selectVideo(section)
        playsound('confirm.wav')
        return section, consecutive_suprises, time.time()



    return section, consecutive_suprises, None




    

def detectHappy(consecutive_happy, section):

    # Increase consecutive smile counter if smiling
    if 'Happy' in pred:
        #print("happy")
        consecutive_happy += 1
        #print(consecutive_smiles)
    else:

        consecutive_happy = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_happy == 20:
        print(f"10 consecutive happy frames detected. Sending MIDI note")

  
        section +=1

        consecutive_happy = 0  # reset the consecutive smile counter
        
        VJ.selectVideo(section)
        playsound('confirm.wav')
        
        return section, consecutive_happy, time.time()
    return section, consecutive_happy, None

def detectAngry(consecutive_angry, section):
    
    # Increase consecutive smile counter if smiling
    if 'Angry' in pred:
       # print("angry")
        consecutive_angry += 1
        #print(consecutive_smiles)
    else:
        # Reset consecutive smile counter if not smiling
        #print("not angry")
        consecutive_angry = 0

    # Sending MIDI signal if 5 consecutive smiles
    if consecutive_angry == 3:
        print(f"10 consecutive angry frames detected. ")

  
        section +=1
        consecutive_angry = 0  # reset the consecutive smile counter
        
        VJ.selectVideo(section)
        playsound('confirm.wav')

        return section, consecutive_angry, time.time()
    return section, consecutive_angry, None


def detectSmileSize(faces, grayscale, img, smile_sizes, section):
    
    mouthGap = mouthOpen(faces, img)
    answers = ["Eeew, what an ugly smile. Tone it down babes.", "You can make that smile bigger, honey. Let your positive energy shine the world!", "You just got the perfect smile!"]
    #using emotion detection (happy) instead of smile detection
    if 'Happy' in pred:
    #if(detectSmile(faces,grayscale, img)):
        
        if(mouthGap > 0.35):
            #print("Eeew, what an ugly smile. Tone it down babes.")
            smile_sizes[0] = smile_sizes[0] + 1
            smile_sizes[1] = 0
            smile_sizes[2] = 0

        elif(mouthGap < 0.2):
            #print("You can make that smile bigger, honey. Let your positive energy shine the world!")
            smile_sizes[0] = 0
            smile_sizes[1] = smile_sizes[1] + 1
            smile_sizes[2] = 0
        else: 
            #print("You just got the perfect smile!")
            smile_sizes[0] = 0
            smile_sizes[1] = 0
            smile_sizes[2] = smile_sizes[2] + 1
        
        print("Consecutive: " + str(max(smile_sizes)))
    else:
        print("You are not smiling and I see it.")
        smile_sizes = [0,0,0]

    if max(smile_sizes) == 7:
        #playsound('confirm.wav')
        
        print(answers[smile_sizes.index(max(smile_sizes))])
      
        section += (1 + smile_sizes.index(max(smile_sizes)))
        VJ.selectVideo(section)

        smile_sizes = [0,0,0]
     
        return section, smile_sizes,  time.time()
    
    return section, smile_sizes,  None



last_time = None
end= False
last_section = 0
last_change_time = time.time()
delays = [43,38,44, 55, 34, 37 ,63, 55, 13, 36, 36, 36, 36]
delays = [4, 13, 36, 36, 36, 36]
stop_event = threading.Event()
emotion_timer = time.time()
#global restart_timer 
#restart_timer = False
def video_processing():
    global section, last_time, end, last_section, last_change_time, consecutive_smiles, pred, emotion_timer, happiness_predictions, current_pred, previous_pred, consecutive_suprises, consecutive_angry, consecutive_happy, smile_sizes
    
    video_getter = VideoGet(0).start()
    
    while not stop_event.is_set():
        
        img = video_getter.frame
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_face.detectMultiScale(grayscale, 1.3, 5)
        
        if(VJ.checkEnds()):
            if(section < 8 and section != 0):
                playsound('error.wav')
            section = 0
            VJ.selectVideo(section)
            """
        try:
            # Check for keyboard input
            if keyboard.is_pressed('n'):
                section+=1
                VJ.selectVideo(section)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C to exit gracefully
            print("Exiting the script.")
            break
        
        if (cv2.waitKey(1) & 0xFF == ord('a')):
            section+=1
            VJ.selectVideo(section)
        """
        for (x, y, w, h) in faces:
            
            if time.time() - emotion_timer >= 1.0:
                fc = grayscale[y:y+h, x:x+w]

                roi = cv2.resize(fc, (48, 48))
                
                # pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                pred_array = model.predict_emotion_array(roi[np.newaxis, :, :, np.newaxis])
                pred = FacialExpressionModel.EMOTIONS_LIST[np.argmax(pred_array)]
                

                current_pred = {emotion: prob for emotion, prob in zip(EMOTIONS_LIST, pred_array[0])}
             
                happiness_predictions.append(current_pred['Happy'])
                
                

                line.set_ydata(happiness_predictions[-100:])
                line.set_xdata(range(len(happiness_predictions[-100:])))
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()

                plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

                cv2.imshow('Happiness Trend', plot_img)

                scoreboard_img = Image.new('RGB', (400, len(EMOTIONS_LIST)*30), (255,255,255)) # create a new white image
                draw = ImageDraw.Draw(scoreboard_img)
                font_scoreboard = ImageFont.truetype("digital-7 (mono).ttf", 15) # load digital font

                emotion_pred_pairs = sorted([(emotion, prob, previous_pred[emotion]) for emotion, prob in current_pred.items()], key=lambda x: x[1], reverse=True)
                
                
                max_emotion, max_prob, _ = emotion_pred_pairs[0]
                
                
                text = f"{max_emotion}: {max_prob:.2f}"
                draw.text((10, 0), text, fill=(255,105,180), font=font_scoreboard) # neon pink
               
                for i, (emotion, prob, _) in enumerate(emotion_pred_pairs[1:], start=1):
                    text = f"{emotion}: {prob:.2f}"
                    draw.text((10, i*30), text, fill='black', font=font_scoreboard)

                
                scoreboard_img_cv = np.array(scoreboard_img) 
                
                
                cv2.imshow('Scoreboard', scoreboard_img_cv)
                
                previous_pred = current_pred
                
                
                emotion_timer = time.time()
            #_, jpeg = cv2.imencode('.jpg', img)

      

        #cv2.putText(img, pred, (x, y), font_emotion, 1, (255, 255, 0), 2)
        if (last_time is None or (time.time() - last_time > delays[section - 1]) and not end):
            """
            if section == 0: #smile to play
               
                #section, consecutive_smiles, last_time = detectConsecutiveSmile(faces, grayscale, img, section, consecutive_smiles)
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
                
            elif section == 1: #smile to continue
             
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
                
                
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
            
            elif section == 8: #be angry"""
            if section == 0:
                #section, consecutive_angry, last_time = detectAngry(consecutive_angry, section)
                section, consecutive_happy, last_time = detectHappy(consecutive_happy, section)
            elif section == 1: #last smile test
                end = True
                section, smile_sizes, last_time = detectSmileSize(faces, grayscale, img, smile_sizes, section)

            
            #case _:
                    
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

