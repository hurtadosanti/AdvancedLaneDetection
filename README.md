# Advanced Lane Detection
![result](results/result.png)

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a threshold binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

You can find a thorough description here on the [detail description](detail_description.md).
 To know what was expected to be accomplished on the project you can review the
  [rubric points](https://review.udacity.com/#!/rubrics/571/view).  

The [Pipeline description](Pipeline.ipynb) shows how the pipeline is used, and the [video pipeline](VideoPipeline.py)
 runs the pipeline on a video stream. To see the implementation of the algorithms look in the
  [utility folder](./utilities).
   Finally, the unit tests and integration tests are located on the [tests folder](./tests) 

## Environment
The environment was created using [miniconda](https://docs.conda.io/en/latest/miniconda.html),
 using the following configurations, [environment.yml](environment.yml).
 
### Libraries
The libraries used on this project are:
  - numpy
  - matplotlib
  - opencv
  - jupyter

 