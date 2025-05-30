import cv2 as cv 
import numpy as np 
# import mediapipe as mp 

#initializing mediapipe hands and drawing stuff 
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils 


#open the webcam 
cap = cv.VideoCapture(0) #0 is the index of the camera, using built in webcam from laptop 

if not cap.isOpened():
    print("Cannot open camera")
    exit() 

while True: 
    #capture video frame by frame 
    ret, frame = cap.read() 

    #dimensions of frame 
    height, width, channels = frame.shape 

    #if the frame is read correctly ret will be True 
    if not ret:
        print("Can't read/receive frame...")
        break 

    #flipping the frame for a mirror effect 
    flipped_frame = cv.flip(frame, 1)
    frame = flipped_frame

    #print(frame.shape) #getting resolution 

    #operations on the frame go here 
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #this method of looping is very slow 
    # for y in range(height): 
    #     for x in range(width):
    #         b, g, r = frame[y, x] #accessing BGR pixel values 
    #         #example: turning bright pixels pure white 
    #         if r > 200 and g > 200 and b > 200: 
    #             frame[y, x] = [0, 0, 0] #white 

    #using numpy is faster and will provide better performance (camera has higher frame rate)
    """
    This line creates a Boolean Mask (a grid of True/False values)
    that identifies which pixels in the image are "bright". i.e where 
    all three color channels are greater than 200 

    so we are comparing each channel value to 200.

    & ensures that all three conditions are True to go forward 

    what is a mask? a mask is a 2D array (height and width) of True/False 

    we can use a mask to filter, modify, or count pixels 
    """
                #[row, column, channel] we are targetting channel where 0 is blue, 1 is green, and 2 is red 
    mask = (frame[:, :, 0] > 200) & (frame[:, :, 1] > 200) & (frame[:, :, 2] > 200)

    """
    Here we are applying the mask to the frame. 

    frame: a NumPy array representing the current video image 

    we are replacing all the pixels where the mask is true with
    pure white in this instance (255, 255, 255)

    so with this line we are saying "for every pixel position where 
    the mask is True, replace the B, G, R values with 255

    """
    frame[mask] = [0, 0, 0] # set all "bright" pixels to black. 


    #draw a diagonal blue line with thickness of 5px, on the frame
    #cv.line(frame,(0,0),(511, 511), (255,0,0), 5) 

    #drawing a rectange on the frame with thickness of 3 
    #cv.rectangle(frame,(384,0),(510,128),(0,255,0),3)

    #cv.rectangle(frame, (300,0), (400,100), (0,255,0), 5)

    """
    !!GOAL!!

    find the biggest mass of black pixels (we've turned all bright pixels to black)

    find the relative center 

    place a box around it (will define this part better, not exactly sure how I want to do it or what it means)

    """

    #STEP 1. 
    """
    To find the biggest black "blob we will need to take the area of all black blobs. Now this is a hard problem to approach I'm realizing becaue
    these blobs aren't always going to be the same shape, this the calculation to find the area will always be different pretty much. Also just
    thought "what if there is a black object" not coming from the light, another thing to handle (maybe change pixel color)

    I need to think of how to approach this...

    I could start with a control object, like a black square not sure. 

    """

    count = 0
    threshold = 10
    #remember this returns a boolean array, so true false. 
    black_pixels = (frame[:, :, 0] <= threshold) & (frame[:, :, 1] <= threshold) & (frame[:, :, 2] <= threshold)# remember the rgb value is accessed in the 3rd element of the image array 

    #this will probably lower the frame rate significantly 
    # for y in range(height):
    #     for x in range(width):
    #         b, g, r = frame[y, x] #getting the rgb values of the pixel 
    #         #print("info b: ", b, "g: ", g, "r: ", r)

    #         if (b <= 10) & (g <= 10) & (r <= 10):
    #             count += 1


    # for x in black_pixels:
    #     if x.any() == True: #get the .any() from an error, it suggested I use .any() or .all() and .any() does the trick. 
    #         count += 1

    count = black_pixels.sum() #WOW ok it's that simple LOL 

    #convert to binary mask 
    """
    black_pixels at this point is a boolean NumPy array as we should know. 

    however OpenCV for the most part doesn't accept boolean arrays for its functions 

    it expects image-like arrays with specific data types and pixel values... 

    findContours() expects an 8bit grayscale image where values expected are:

        - 0 for background 
        - 255 for the blobs (white areas) 

    so the conversion we do with black_pixels.astype(np.uint8) * 255 

        takes the initial 1's and 0's from the binary numpy array and converts them to what we need 

        *255 makes it so 0's are still 0's 
        but 1 -> 255

        so now we have 255 where black pixels are
        and 0 for everywhere else (the background)

    in the end we create a valid binary image for cv.findContours() to work with. 

    ------------
    INITIAL ERROR: 
    Before when I wasn't using that I was running into this error: 

        image data type = bool is not supported

    and that's because we were still using bool values in the function,
    after conversion we were good. 

    """
    black_mask = black_pixels.astype(np.uint8) * 255 #the conversion 

    #figure out how to differentiate between "black blobs/masses" -- contours and functions below determine that for us as it can differentiate the differences between pixels i.e the contours 
    #we need to use something called Contour Detection! 
    #contours let us detect seperate shapes or masses of a specific color. 

    #so lets find the contours (the , _ is there because findContours returns a few different values but I only care about the first one which are the list of contours each is a NumPy array of points) SO WHAT THIS MEANS is that , _ is a Pythonic way to ignore the value 
    contours, _ = cv.findContours(black_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours: 
        area = cv.contourArea(contour) #calculate the area based on our contour. 
        if area > 500: #we are ignoring small black blobs 
            #get the bounding box 
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #draw the center 
            # M = cv.moments(contour)
            # if M['m00'] != 0:
            #     cx = int(M['m10']/M['m00'])
            #     cy = int(M['m01']/M['m00'])
            #     cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1) 


    print("Black Pixels: ", count)

    #display the resulting frame 
    cv.imshow('Manual Pixel Processing', frame)
    if cv.waitKey(1) == ord('q'): 
        break 

#when everything is done, release the capture 
cap.release()
cv.destroyAllWindows() 