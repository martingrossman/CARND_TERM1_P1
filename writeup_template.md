# **Finding Lane Lines on the Road** 



---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on work in a written report

---

### Reflection

### 1. Task - introduction

The goal to detect lines in the video is solved using classical image processing techniques. I have used OpenCV library for most operations responsible for image processing.

Task can be divided into few simple steps.

1. open video file
2. read the frame
3. process each frame of video files and detect lines on the road
4. create a new clip with the detect lines

### 2. Processing of the frames

Processing of each individual frame is the core of this project. It can be divided into:

1. Applying series of filters which produce black&white image with detected edges - **IPFilters()**
   1. selecting only white and yellow colors (line colors)
   2. conversion to grayscale
   3. gaussian blur
   4. canny edge detection
   5. selecting only region of interest
2. Detecting straight lines using Hough transform trick  and converting these lines into final Left and Right Line **hough_lines()**
   1. Hough transform - detection of all straight lines
   2. sorting into left, right lines
   3. checking for min, max slope
   4. averaging separate left and right lines
      - average of start points and average of end points. (see section averaging)
   5. smoothing resulting lines with previous lines from last frame
      - To stabilize lines: convex combination of the x coordinates of the current line and previously detected line
   6. drawing lines on black image

The program flow chart is depicted below:

!['flow diagram'](C:\Users\martin_dev\Documents\UDACITY_ND\CarND-LaneLines-P1\documentation\flow_1.png)



![](C:\Users\martin_dev\Documents\UDACITY_ND\CarND-LaneLines-P1\documentation\hough_lines.png)



### 2.Averaging of lines

On internet I saw many people are averaging slopes of individual lines to average lines. This is in principle not correct. As line average is not mathematically defined I assume the output of such function should provide for two lines with different slopes a line with slope at which the new line divides angle of the input lines into half. Averaging of different slopes doesn't provide such result. (e.g. if line1 has slope >> 0 while line 2 has slope = 1, their average is still very high (k1+k2)/2 >>0 ) Therefore I decided to average x,y of start points and end points of the lines separately. Please see image below.

![](C:\Users\martin_dev\Documents\UDACITY_ND\CarND-LaneLines-P1\documentation\LineAveragingDisimilar.png)

![](C:\Users\martin_dev\Documents\UDACITY_ND\CarND-LaneLines-P1\documentation\LineAveragingSimilar.png)




### 2. Identify potential shortcomings with your current pipeline

There are many shortcomings. Most obvious for me are:

- a lot of manual tuning of the parameters and still in many situations probably  it wont work that great or not at all
- the algorithm doesn't have any adaptation to light changes
- algorithm as it is now, doesn't know by himself how 'sure' he is about detection
- doesn't take into account curvature of the lines
- doesn't take into account camera and perspective distortion
- from software point of view it could be written less memory hungry


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to:

- compute vanishing point to adjust ROI
- take into account ego motion
- more advanced adaptive thresholding
- instead of LowPass use Kalman 