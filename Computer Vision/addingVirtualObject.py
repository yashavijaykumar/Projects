# -*- coding: utf-8 -*-
import cv2
import numpy as np
def mouse_handling(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points_from_image(im):
    
    # Set up data to send to mouse handling event
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handling, data)
    cv2.waitKey(0)
    
    points = np.vstack(data['points']).astype(float)
    
    return points


if __name__ == '__main__' :

    # Read source image.
    virtualObj = cv2.imread('UD.png');
    size = virtualObj.shape
   
    # Create a vector of source points.
    pts_src = np.array(
                       [
                        [0,0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] -1],
                        [0, size[0] - 1 ]
                        ],dtype=float
                       );
    
     # Read frame image 
    image_frame = cv2.imread('1.jpg');

    # Get four corners 
    print ('Click on four points where you want to place the virtual object and then press ENTER')
    pts_dst = get_four_points_from_image(image_frame)
    
    # Calculate Homography between source and destination points
    h, status = cv2.findHomography(pts_src, pts_dst);
    
    # Warp virtual object image
    im_temp = cv2.warpPerspective(virtualObj, h, (image_frame.shape[1],image_frame.shape[0]))

    cv2.fillConvexPoly(image_frame, pts_dst.astype(int), 0, 16);
    
    # Add warped source image to destination image.
    image_frame = image_frame + im_temp;
    
    # Display image.
    cv2.imshow("Image", image_frame);
    k = cv2.waitKey(0)
    if k == 27:                          # wait for ESC key to exit
       cv2.destroyAllWindows()
    elif k == ord('s'):                 # wait for 's' key to save and exit
       cv2.imwrite('17.jpg',image_frame)
       cv2.destroyAllWindows()