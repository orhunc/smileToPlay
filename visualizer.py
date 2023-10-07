import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps

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
  #  plot_img = ImageOps.expand(plot_img, border=3, fill='black')

    # Define the border size in pixels
    border_size = 3
    """
    # Create a black canvas with a border
    canvas_height, canvas_width, _ = plot_img.shape
    bordered_canvas = np.zeros((canvas_height + 2 * border_size, canvas_width + 2 * border_size, 3), dtype=np.uint8)
    bordered_canvas[border_size:-border_size, border_size:-border_size] = plot_img

    # Draw the black border
    
    bordered_canvas[:border_size, :] = (0, 0, 0)
    bordered_canvas[-border_size:, :] = (0, 0, 0)
    bordered_canvas[:, :border_size] = (0, 0, 0)
    bordered_canvas[:, -border_size:] = (0, 0, 0)"""

    # Define the border color (yellow in BGR format)
    border_color =  (0, 237, 255)#(0,226,254)(254,226,0)#(0, 255, 255)  # Yellow in OpenCV's BGR color format (255, 237, 0)

    # Create a canvas with the border color
    canvas_height, canvas_width, _ = plot_img.shape
    canvas_with_border = np.ones((canvas_height + 2 * border_size, canvas_width + 2 * border_size, 3), dtype=np.uint8)
    canvas_with_border[:] = border_color

    # Place the plot image on the canvas
    canvas_with_border[border_size:-border_size, border_size:-border_size] = plot_img
    plot_img = canvas_with_border

    #emotions scoreboard
    scoreboard_img = Image.new('RGB', (110, len(EMOTIONS_LIST)*30), (245,246,249)) # create a new white image (245,246,249)
    scoreboard_img = ImageOps.expand(scoreboard_img, border=3, fill=(0, 237, 255))
    draw = ImageDraw.Draw(scoreboard_img)
    font_scoreboard = ImageFont.truetype("digital-7 (mono).ttf", 15) # load digital font

    emotion_pred_pairs = sorted([(emotion, prob, previous_pred[emotion]) for emotion, prob in current_pred.items()], key=lambda x: x[1], reverse=True)
    max_emotion, max_prob, _ = emotion_pred_pairs[0]
    
    
    text = f"{max_emotion}: {max_prob:.2f}"
    draw.text((10, 10), text, fill=(255,105,180), font=font_scoreboard) # neon pink
    
    for i, (emotion, prob, _) in enumerate(emotion_pred_pairs[1:], start=1):
        text = f"{emotion}: {prob:.2f}"
        draw.text((10, (i*30)+10), text, fill='black', font=font_scoreboard)

    
    scoreboard_img_cv = np.array(scoreboard_img) 
    previous_pred = current_pred
    return plot_img, scoreboard_img_cv


def showSmileQuality(section, fakeness, smileQuality):
        #emotions scoreboard
        #smile_scoreboard_img = Image.new('RGB', (320, 52), (0, 237, 255)) # create a new white image
        #smile_scoreboard_img = Image.new('RGB', (170, 52), (0, 237, 255)) # create a new white image
        smile_scoreboard_img = Image.new('RGB', (320, 32), (0, 237, 255)) # create a new white image

        #smile_scoreboard_img = ImageOps.expand(smile_scoreboard_img, border=3, fill=(0,226,254))
        draw = ImageDraw.Draw(smile_scoreboard_img)
        font_scoreboard = ImageFont.truetype("digital-7 (mono).ttf", 15) # load digital font
        tmp =5
        #if section > 9:
        text = "Fakeness: " + str(fakeness) + "%"  +"         Smile Quality: " + str(smileQuality) + "/100"

        #text = "Fakeness: " + str(fakeness) + "%" + "\n" +"Smile Quality: " + str(smileQuality) + "/100"
        #text = "Your smile evaluation:" + "\n" +"Fakeness: " + str(fakeness) + "%"   +"          Smile Quality: " + str(smileQuality) + "/100"
        #else: 
            #text = "Fakeness: " + str(fakeness) + "%" + "\n" +"Smile Quality: " + str(smileQuality) + "/100"

     
        draw.text((10, 10), text, fill='black', font=font_scoreboard) 
        
        
        smile_scoreboard_img_cv = np.array(smile_scoreboard_img) 
        return smile_scoreboard_img_cv

def noFace():
    global previous_pred
    current_pred = {'Angry': 0.000, 'Disgust': 0.00, 'Fear': 0.00, 'Happy': 0.00, 'Neutral': 0.00, 'Sad': 0.00, 'Surprise': 0.00}
    scoreboard_img = Image.new('RGB', (110, len(EMOTIONS_LIST)*30), (245,246,249)) # create a new white image
    scoreboard_img = ImageOps.expand(scoreboard_img, border=3, fill=(0,226,254))

    draw = ImageDraw.Draw(scoreboard_img)
    font_scoreboard = ImageFont.truetype("digital-7 (mono).ttf", 15) # load digital font

    emotion_pred_pairs = sorted([(emotion, prob, previous_pred[emotion]) for emotion, prob in current_pred.items()], key=lambda x: x[1], reverse=True)
    max_emotion, max_prob, _ = emotion_pred_pairs[0]
    
    
    text = f"{max_emotion}: {max_prob:.2f}"
    draw.text((10, 10), text, fill=(255,105,180), font=font_scoreboard) # neon pink
    
    for i, (emotion, prob, _) in enumerate(emotion_pred_pairs[1:], start=1):
        text = f"{emotion}: {prob:.2f}"
        draw.text((10, (i*30)+10), text, fill='black', font=font_scoreboard)
   
    

    scoreboard_img_cv = np.array(scoreboard_img) 
    previous_pred = current_pred
    return scoreboard_img_cv
    
   
