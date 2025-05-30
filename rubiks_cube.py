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

    #NEXT WORK GOES IN HERE 
    #----------------------

    """
    ok I think I'll need to make a filter for each color of the cube so lets start with that 
    """
    redthresh = 200
    bluethresh = 150
    greenthresh = 230
    whitethresh = 150 #(255, 255, 255)
    orangethresh, orangethresh1 = 230, 140  #(255, 165, 0)
    yellowthresh = 180 #(255, 255, 0)

    #------------MASKS------------ 
    #REMEMBER: channels - 0 = blue   |  1 = green   |  2 = red  (it's flipped, BGR)
    red = (frame[:, :, 0] <= 100) & (frame[:, :, 1] <= 100) & (frame[:, :, 2] > redthresh) 
    blue = (frame[:, :, 0] > bluethresh) & (frame[:, :, 1] <= 100) & (frame[:, :, 2] <= 100) 
    green = (frame[:, :, 0] <= 0) & (frame[:, :, 1] > greenthresh) & (frame[:, :, 2] <= 0) 
    white = (frame[:, :, 0] > whitethresh) & (frame[:, :, 1] > whitethresh) & (frame[:, :, 2] > whitethresh) 
    orange = (frame[:, :, 0] <= 0) & (frame[:, :, 1] > orangethresh1 ) & (frame[:, :, 1] < 190) & (frame[:, :, 2] > orangethresh) 
    yellow = (frame[:, :, 0] <= 100) & (frame[:, :, 1] > yellowthresh) & (frame[:, :, 2] > yellowthresh) 


    #moved this lower I don't think it's an issue but will leave this here INCASE 
    # #set pixels for better contours 
    # frame[blue] = [255, 0, 0] #setting blue pixels to blue 


    #now I think I have to do conversions for all these as they are returning boolean NumPy arrays atm
    red_mask = red.astype(np.uint8) * 255 
    blue_mask = blue.astype(np.uint8) * 255 
    green_mask = green.astype(np.uint8) * 255
    white_mask = white.astype(np.uint8) * 255
    orange_mask = orange.astype(np.uint8) * 255
    yellow_mask = yellow.astype(np.uint8) * 255

    #now that we have converted we should be able to detect contours 
    redContours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blueContours, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    greenContours, _ = cv.findContours(green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    whiteContours, _ = cv.findContours(white_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    orangeContours, _ = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    yellowContours, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #debug 
    # redpixels = red.sum()
    # print("red: ", redpixels)
    # bluepixels = blue.sum()
    # print("blue: ", bluepixels)
    yellowpixels = yellow.sum()
    print("yellow: ", yellowpixels)

    """
    ok the below is actually helping me in a way I didn't expect, since we are turning the pixels that fall 
    within my range of what a blue pixel is, and then taking that blue pixel and setting it to straight blue RGB(0,0,255)
    we can visualize what is being picked up by the camera and our range as blue.

    I can actually see whats happening. 

    What I've learned? 
    it seems light is definitley one of the bigger issues here (I've been testing in nanny and poppas basement with minimal light)
    """
    #set pixels so I can see visually what's getting picked up. 
    frame[blue] = [255, 0, 0] 
    frame[red] = [0, 0, 255]
    frame[yellow] = [0, 255, 255]
    frame[white] = [255, 255, 255]

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

    """
    ok I'm not sure what the exact reason is yet, but finding the center of the things I'm trying to track, or at least the bounding
    box itself is just not getting put around the objects I hold up
        now this could be because of a number of factors like my area condition, lighting irl, the objects I'm using to test
        are just not good for this, etc. 

    either way I had an idea. What if when we got the color we need we overlaid it with a solid blue color for instance, then it
    would stand out more and be easier to track the contours... maybe?? 

    because when putting the bounding box around the black pixels before they seemed pretty good at getting around the area 
    of black pixels. They were "off centered" or not around the black pixels, they were more accurate (seemingly)

    so this could be a good try at least. 

    steps: 
        - get the target color
        - replace it with a bolder color 
        - put contours around that 
    """



    #----------------------
    cv.imshow('Rubiks Cube Sovler', frame)
    if cv.waitKey(1) == ord('q'):
        break #close program d

#when everything is done, release the capture 
cam.release()
cv.destroyAllWindows() 
