import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


#class Depicter{}
import time



global previous_pred, fig, ax, line, canvas
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
previous_pred = {emotion: 0 for emotion in EMOTIONS_LIST}



happiness_predictions = []

def init():
    global fig, ax, line, canvas
    sns.set()
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    line, = ax.plot([], [], color='hotpink', linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Happiness Trend")
    canvas = FigureCanvas(fig)

def main(pred_array):
    global previous_pred

    #happiness chart
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



    #emotions scoreboard
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
    previous_pred = current_pred
    return plot_img, scoreboard_img_cv


def showSmileQuality(section, fakeness, smileQuality):
        #emotions scoreboard
        smile_scoreboard_img = Image.new('RGB', (400, 100), (255,255,255)) # create a new white image
        draw = ImageDraw.Draw(smile_scoreboard_img)
        font_scoreboard = ImageFont.truetype("digital-7 (mono).ttf", 15) # load digital font
        tmp =5
        if section > 9:
            text = "Fakeness: " + str(fakeness) + "%" + "\n" +"Smile Quality: " + str(smileQuality) + "/100"
        else: 
            text = ""

        print(text)
        draw.text((10, 0), text, fill='black', font=font_scoreboard) 
        
        
        smile_scoreboard_img_cv = np.array(smile_scoreboard_img) 
        return smile_scoreboard_img_cv

def noFace():
    global previous_pred
    current_pred = {'Angry': 0.000, 'Disgust': 0.00, 'Fear': 0.00, 'Happy': 0.00, 'Neutral': 0.00, 'Sad': 0.00, 'Surprise': 0.00}
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
    previous_pred = current_pred
    return scoreboard_img_cv
    
   
