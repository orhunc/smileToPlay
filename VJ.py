import pygame
import os
from ffpyplayer.player import MediaPlayer
from ffpyplayer.tools import set_loglevel
from pymediainfo import MediaInfo
from errno import ENOENT
from pyvidplayer import Video
import time


input = False
restart = False 

def selectVideo(section):
    global input   
    global current_video_index
    current_video_index = section
    input = True
"""
def restart():
    global restart
    restart = True
"""
video_files = ["STP01_intro.mp4", "STP02.mp4", "STP03.mp4", "STP04.mp4", "STP05.mp4", "STP06.mp4", "STP07.mp4", "STP08.mp4", "STP09.mp4", "STP10.mp4", "STP11_C.mp4", "STP11_B.mp4", "STP11_A.mp4","STP11_A.mp4"]
global current_video_index, current_video_file, vid 


def play_pause():
    global vid
    vid.toggle_pause()

def checkEnds():
    global current_video_index, input
    if(vid.has_ended()):
            #vid.restart() 
            current_video_index = 0
            input = True
            return True
    return False

def main():
    global input, video_files, current_video_file, current_video_index, vid
    pygame.init()
    win = pygame.display.set_mode((1920, 1080))
    clock = pygame.time.Clock()

    video_files = ["STP01_intro.mp4", "STP02.mp4", "STP03.mp4", "STP04.mp4", "STP05.mp4", "STP06.mp4", "STP07.mp4", "STP08.mp4", "STP09.mp4", "STP10.mp4", "STP11_C.mp4", "STP11_B.mp4", "STP11_A.mp4","STP11_A.mp4"]  # Replace with your video files
    #video_files = ["STP01_intro.mp4",  "STP11_C.mp4", "STP11_C.mp4" ,  "STP11_C.mp4",  "STP11_C.mp4" ,  "STP11_C.mp4" ,  "STP11_C.mp4", "STP11_C.mp4",  "STP11_C.mp4", "STP11_C.mp4",  "STP11_C.mp4","STP11_B.mp4", "STP11_A.mp4"]  # Replace with your video files

    current_video_index = 0
    current_video_file = video_files[current_video_index]
    vid = Video(current_video_file)

    while True:
        #time.sleep(1)
        key = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                vid.close()
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)

        # Your program frame rate does not affect video playback
        clock.tick(50)
        
        
        
        if key == "r":
            vid.restart()  # Rewind video to the beginning
            
        elif key == "p":
            vid.toggle_pause()  # Pause/plays video
        elif key == "right":
            vid.seek(15)  # Skip 15 seconds in the video
        elif key == "left":
            vid.seek(-15)  # Rewind 15 seconds in the video
        elif key == "up":
            vid.set_volume(1.0)  # Max volume
        elif key == "down":
            vid.set_volume(0.0)  # Min volume
        elif key == "n" or input:
            # Switch to the next video
            vid.close()
            current_video_index = (current_video_index ) % len(video_files)
          
            #gpt_v6.section = current_video_index
            current_video_file = video_files[current_video_index]
           
            vid = Video(current_video_file)
            
            input = False

        # Draws the video to the given surface, at the given position
        vid.draw(win, (0, 0), force_draw=False)

        #vid.printer()
       #print(vid.has_ended())
        pygame.display.update()
        



