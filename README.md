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

## Detail Description

- The [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) describe what is expected for the project 

- A [Detailed Report](Report.md) presents what was done 

- The [Pipeline description](Pipeline.ipynb) shows how the pipeline is used

- The [video pipeline](video_pipeline.py) runs the pipeline for a [video](https://youtu.be/yAzrk6jL2NY)  

- On the [utilities folder](./utilities) can be found the complete implementation of the algorithms

- Unit tests and integration tests are located on the [Test folder](./tests) 

## Environment
The environment was created using [miniconda](https://docs.conda.io/en/latest/miniconda.html),
 using the following configurations:

### Channels
  - defaults
### Dependencies
  - numpy
  - matplotlib
  - opencv
  - jupyter

 The complete environment can be found on the [yml file](environment.yml) 