# Recommended read
* [Highly Curved Lane Detection AlgorithmsBased on Kalman Filter](https://www.mdpi.com/2076-3417/10/7/2372/htm)
* [Lane Detection Based on Road Module and Extended Kalman Filter](https://www.researchgate.net/publication/323197197_Lane_Detection_Based_on_Road_Module_and_Extended_Kalman_Filter)

* [Tutorial: Build a lane detector - CNN](https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132)

* [A Fast Learning Method for Accurate and Robust Lane Detection Using Two-Stage Feature Extraction with YOLO v3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6308794/)

* [Robust lane detection and object tracking](./FULLTEXT01.pdf)
***
# Camera calibration

* [camera_calibration](http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html?)
* [camera_calibration_and_3d_reconstruction](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
* [tutorial_py_calibration](http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html)
* [Microsoft calibration](http://research.microsoft.com/en-us/um/people/zhang/calib/)
***
# Color Filtering

* HLS and HSV.
    * H channel gives extremely noisy output and S channel of HLS gives better results which you done it. On the other side, HSV's V channel gives clear grayscale image, i.e the yellow line, much better than L channel.
* LAB, RGB and LUV
    * Along with color threshold to the B(range:145-200) in LAB for shading & brightness applying it to R in RGB in final pipeline can also help in detecting the yellow lanes.
    * And thresholding L (range: 215-255) of LUV for whites.


# Methods for lane pixel extraction

* [Lane detection method 1](https://www.researchgate.net/publication/257291768_A_Much_Advanced_and_Efficient_Lane_Detection_Algorithm_for_Intelligent_Highway_Safety)
* [Lane detection method 2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017478/)
* [Algorithm based on current project:](https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa#.l2uxq26sn)

# Pipeline Video Suggestion

In addition to other filtering mechanisms. You can also use cv2.matchShapes as a means to make sure the final warp polygon is of good quality. This can be done by comparing two shapes returning 0 index for identical shapes. You can use this to make sure that the polygon of your next frame is closer to what is expected and if not then can use the old polygon instead. This way you are faking it until a new frames appear and hence will get good results.