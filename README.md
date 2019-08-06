# HUMAN DETECTION USING AWR1642 & YOLOv3
Object detection system is one of the current needs. The increasing use of object detection systems in various fields has resulted in the fast development of object detection systems. The development carried out according to the needs or needs of each case example. Current conditions require the development of a reliable object detection system. One of the most reliable object detection systems is an object detection system that is independent with light intensity. Therefore in this Final Project an object detection system is made using radar and cameras to be able to detect without the dependence of light intensity. Object detection systems are made to be able to detect human classes, so that the output of this system is the number of humans detected. Object detection system using a camera used artificial intelligence to improve the performance using the YOLO model, while for the object detection system using radar, the AWR1642 was used. The detection results of this object detection system are finally displayed on the web interface

<video width="700px" height="500px" src="https://youtu.be/0CkqzWnJWqA"></video>

For object detection system we made, we split the system into 3 subsystem:
<ul>
  <li>AWR System</li>
  <li>YOLO System</li>
  <li>WEBSITE System</li>
</ul>
<img src="https://github.com/STEI-ITB/AMA/blob/master/final%20sistem.png?raw=true"></img>

<h1>AWR System</h1>
<img src="https://github.com/STEI-ITB/AMA/blob/master/AWR1642.jpg?raw=true"></img>
This system used AWR1642 for processing signal reflection and transform it into datapoints. from here we are using DBSCAN clustering for group the datapoints as human. Here the steps for using AWR System :
<ul>
  <li>Plug in AWR1642 then flash with mmwaveSDK 2.0(http://software-dl.ti.com/ra-processors/esd/MMWAVE-SDK/lts-latest/index_FDS.html)</li>
  <li>Run code_awr.py</li>
  <li>Choose COM PORTS(Number of ports depend on your PC, check in Device Manager)</li>
  <li>In "Real Time Tunning" section uncheck "Range Detection" & "Doppler Direction" for better clustering and check "Remove Static Clutter" for remove any static object reflection</li>
</ul>


<h1>YOLO System</h1>
<img src="https://github.com/STEI-ITB/AMA/blob/master/yolo.png?raw=true" width="400px"></img>
YOLO system used any image or video for the input. In this case we are using external webcam for the input. here the steps for using YOLO System :
<ul>
  <li>Plug in external webcam</li>
  <li>Run code_awr.py</li>
  <li>Choose COM PORTS(Number of ports depend on your PC, check in Device Manager)</li>
</ul>
if you are using notebook webcam, change "1" to "0" in line 126 "code_yolo.py" as internal webcam. Also if you want change the input from camera to RSTP stream or video file or image file, you can change the variable by uncomment "cap" in line 128 or 124 "code_yolo.py" .


<h1>WEBSITE System</h1>
<img src="https://github.com/STEI-ITB/AMA/blob/master/Website.PNG?raw=true"></img>
WEBSITE system captured data stream using AJaX. Then this system need web server for GET method. We are using Flask for this purpose. AJaX only search and get the data in "static" folder. make sure your output file.csv and file.png in this folder. here the steps for using WEBSITE system :
<ul>
  <li>Run RunFlask.py</li>
  <li>Go to local host (http://127.0.0.1:5000)</li>
</ul>
