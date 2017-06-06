## Project: Search and Sample Return


**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./misc/roverview.png
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded).
Add/modify functions to allow for color selection of obstacles and rock samples.

I created the following functions for navigable terrain, rocks (obstacles) and Sample Rocks:

```python


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 255
    # Return the binary image
    return color_select

threshed = color_thresh(warped)
plt.imshow(threshed, cmap='gray')
#scipy.misc.imsave('../output/warped_threshed.jpg', threshed*255)

#identify pixels of rocks - obstacles
def color_thresh_rock(img, hsv_thresh_lower=(20, 100, 100), hsv_thresh_upper=(30, 255, 255)):
    color_select = np.zeros_like(img[:,:,0])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #modify the upper and lower bounds of the filter

    lower_gold = np.array([hsv_thresh_lower[0], hsv_thresh_lower[1], hsv_thresh_lower[2]])
    upper_gold = np.array([hsv_thresh_upper[0], hsv_thresh_upper[1], hsv_thresh_upper[2]])

    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    res = cv2.bitwise_and(img,img, mask= mask)
    color_select[mask] = 255

    return color_select

#identify pixels of the gold rocks
#TODO - make this work better
def color_thresh_obs(img, rgb_thresh_lower=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    thresh =      (img[:,:,0] < rgb_thresh_lower[0]/3)  \
                & (img[:,:,1] < rgb_thresh_lower[1]/3) \
                & (img[:,:,2] < rgb_thresh_lower[2]/3)
    # Index the array of zeros with the boolean array and set to 1
    color_select[thresh] = 255
    # Return the binary image
    return color_select
```
The functions above give us these results:

![alt text][image2]



#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
The completed process_image() function:

```python
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # TODO: 
    # 1) Define source and destination points for perspective transform
    src_pts = np.float32([[14, 142], [302, 142], [200, 97], [118,96]])
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6

    dst_pts = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    # 2) Apply perspective transform
    warped = perspect_transform(img, src_pts, dst_pts)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    binary = color_thresh(warped)
    threshedRock = color_thresh_rock(warped)
    threshedObs = color_thresh(warped, (165,165,165))

   
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    xpix, ypix = rover_coords(binary)
    xpixRock, ypixRock = rover_coords(threshedRock)
    xpixObs, ypixObs = rover_coords(threshedObs)
    
    # 5) Convert rover-centric pixel values to world coords
    w_xpix, w_ypix = to_polar_coords(xpix, ypix)
   
    # 6) Update worldmap (to be displayed on right side of screen)
    

    
    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Co-ords X: {}, Y: {}".format(data.xpos[data.count], data.ypos[data.count]), (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image

```


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
The `perception_step()` function is very similar to the `process_image()` function from the Jupyter notebook.  I have included it here for reference

```python
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    src_pts = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    dst_pts = np.float32([[Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                              [Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                              ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, src_pts, dst_pts)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshold = color_thresh(warped)
    threshedRock = color_thresh_rock(warped)
    threshedObs = color_thresh(warped, (165,165,165))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = threshedObs
    Rover.vision_image[:,:,1] = threshedRock
    Rover.vision_image[:,:,2] = threshold

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshold)
    xpixRock, ypixRock = rover_coords(threshedRock)
    xpixObs, ypixObs = rover_coords(threshedObs)


    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = pix_to_world(xpixObs, ypixObs, Rover.pos[0], Rover.pos[1], Rover.yaw,
                                                      Rover.worldmap.shape[0], 10)
    rock_x_world, rock_y_world = pix_to_world(xpixRock, ypixRock, Rover.pos[0],
                                              Rover.pos[1], Rover.yaw,
                                              Rover.worldmap.shape[0], 10)
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, Rover.pos[0],
                                                        Rover.pos[1], Rover.yaw,
                                                        Rover.worldmap.shape[0], 10)


    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    dist, angles = to_polar_coords(xpix, ypix)
    distRock, anglesRock = to_polar_coords(xpixRock, ypixRock)

    distObs, anglesObs = to_polar_coords(xpixObs, ypixObs)
    mean_dirObs = np.mean(anglesObs)

    Rover.nav_dists = dist
    Rover.nav_angles = angles
    #Rover.rock_dists = distRock
    #Rover.rock_angles = anglesRock
    

    return Rover

```

unlike the jupyter notebook that uses a class called `DataBucket` to simulate the Rover object, the `perception_step()` function has full access to the Rover object

we are able to update the ground_truth map to show:
 <ul>
 <li>where the rover has been (navigable_x_world, navigable_y_world)</li>
 <li>where the samples are (rock_y_world, rock_x_world)
 </ul>

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

Simulator Settings
<ul>
<li>Resolution: 1024 X 768 </li>
<li>Graphics Quality: Good </li>
<li>FPS: 14 </li>
</ul>


**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  
Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) 
in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I essentially followed the same pipeline as the lesson and Jupyter notebooks. The result was that in Autonomous mode the rover was able to map 40.3% of the map with a fidelity of 71%

The video of autonomous mode can be seen here [here](https://www.youtube.com/watch?v=u19ApOK0FL0&feature=youtu.be)

The pipeline (repeated for each frame):
<ul>
<li>I added the thresholding functions to the perception.py file to return binary images of obstacles, navigable terrain and sample rocks</li>
<li>I then performed a prespective transform on the binary images</li>
<li>With a perspective tranform complete I converted the rover-centric coordinates to "realworld coordinates"</li>
<li>We also apply a rotation and transformation to bring the x,y pos of the rover to (0,0) on the map.</li>
<li>We calculate the mean angle - which guides the rover</li>
<li>Finally, we update the Rover object with fresh settings
</ul>

This pipeline is rudimentary and due to a lack of time on my part - I was not able to complete the extra challenge to detect and pick up sample rocks.

Some other things I would like to go back and change:
<ul>
<li>No mechanism to record the already visited coordinates and explore the entire map before re-visiting already explored coords</li>
<li>In one instance, the mean of the angle was such that the rover was stuck going in a large circle in the middle of the map</li>
<li>This version does not track sample rocks very well.</li>
</ul>


