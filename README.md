# object_distance_measurement_and_recognition



This project leverages Intel RealSense technology to create a comprehensive solution for measuring an object's distance from the camera,
identifying the object, and generating a mask for precise isolation.

Key Features:

Distance Measurement: Utilizing Intel RealSense depth sensing capabilities, this project provides accurate distance measurements 
between the camera and the target object. The RealSense SDK is employed to access depth data and calculate distances, ensuring reliable and precise results.

Object Detection: Using object detection algorithms, the system identifies and classifies objects within the camera's field of view.
This functionality enhances the project's versatility, allowing it to recognize a wide variety of objects.

Mask Generation: The repository includes modules for creating masks that precisely outline the detected objects.
These masks enable easy extraction and manipulation of object-specific data, facilitating downstream processing and analysis.

OpenCV Integration: The project seamlessly integrates with the OpenCV library to enhance image processing capabilities. 
OpenCV is employed for various tasks, including image manipulation, contour detection, and mask generation.
