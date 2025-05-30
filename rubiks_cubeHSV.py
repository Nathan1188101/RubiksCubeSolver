"""
Going to see if I can make an app that will solve a rubiks cube

GOAL FOR NOW: 
    the first step I need to start with is at least getting bounding boxes around each little square on one side

ULTIMATE GOAL:
    solve rubiks cube 

IDEAS FOR METHOD: 
    
    - need to identify the 6 sides of the cube, we can have error handling to prevent the user from scanning the same side twice
      by storing each side in mem and then comparing each new side compared to what we have in mem. 

    - lighting will be hard to control so there will have to be a small range for each color to get accepted for tracking. 

    - each rubiks cube isn't made the same either so that range will help as well as colors won't always be consitent. 

    - will need to learn the algorithm to solving a rubiks cube, the "shortest path" version of that if it exists.

    - will need to figure out how to give the solution to the user 

"""

import cv2 as cv 
import numpy as np

#gotta start by opening the camera device 
cam = cv.VideoCapture(0) #whatever default is

#setting the camera dimensions to 720p (what my webcam is)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True: 
    #capture video frame by frame 
    ret, frame = cam.read() 

    #dimensions of the frame 
    height, width, channels = frame.shape 

    #if the frame is read correctly ret will be TRUE 
    if not ret:
        print("Can't read/receive frame...")
        break

    #flip the frame so it is oriented correctly 
    flipped_frame = cv.flip(frame, 1)
    frame = flipped_frame 

    #this converts my "main" frame
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #converting the frame to HSV for better color tracking 

    #having a variable of HSV conversion serperatley 
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #NEXT WORK GOES IN HERE 
    #----------------------

    #HSV COLOR RANGES 
    # Red (two ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Blue
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Green
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Orange
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Yellow
    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # White
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    #MASKS 
    red_mask = cv.inRange(hsv, lower_red1, upper_red1) | cv.inRange(hsv, lower_red2, upper_red2)
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv.inRange(hsv, lower_green, upper_green)
    orange_mask = cv.inRange(hsv, lower_orange, upper_orange)
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv.inRange(hsv, lower_white, upper_white)

    #now that we have converted we should be able to detect contours 
    redContours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blueContours, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    greenContours, _ = cv.findContours(green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    whiteContours, _ = cv.findContours(white_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    orangeContours, _ = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    yellowContours, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #debug 
    red_pixels = np.count_nonzero(red_mask)
    print("red: ", red_pixels)

    """
    ok the below is actually helping me in a way I didn't expect, since we are turning the pixels that fall 
    within my range of what a blue pixel is, and then taking that blue pixel and setting it to straight blue RGB(0,0,255)
    we can visualize what is being picked up by the camera and our range as blue.

    I can actually see whats happening. 

    What I've learned? 
    it seems light is definitley one of the bigger issues here (I've been testing in nanny and poppas basement with minimal light)
    """
    #set pixels so I can see visually what's getting picked up. 
    # frame[blue] = [255, 0, 0] 
    # frame[red] = [0, 0, 255]
    # frame[yellow] = [0, 255, 255]
    # frame[white] = [255, 255, 255]

    #NEED TO ORGANIZE THIS STUFF BETTER THE CONTINUITY OF MY IDEAS IS GOING TO BE HARD TO COME BACK TO AND READ I'M ALL OVER THE PLACE 
    #now we should be able to loop through and begin calculations and drawing for tracking 
    #think I will need a loop for each color 
    for contour in redContours:
        #get the area based on the contour 
        area = cv.contourArea(contour)
        #will need to tweak this to better capture the squares on the cube
        #for now
        if area > 100:
            #get bounding box 
            x, y, w, h = cv.boundingRect(contour)
            #had this wrong for a while, thats why the boxes were all being drawn off center
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for contour in blueContours:
        area = cv.contourArea(contour)
        if area > 100:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for contour in yellowContours:
        area = cv.contourArea(contour)
        if area > 100:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for contour in whiteContours:
        area = cv.contourArea(contour)
        if area > 100: 
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)




    #----------------------
    cv.imshow('Rubiks Cube Sovler', frame)
    if cv.waitKey(1) == ord('q'):
        break #close program d

#when everything is done, release the capture 
cam.release()
cv.destroyAllWindows() 
