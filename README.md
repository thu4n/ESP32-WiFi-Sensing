# ESP32-WiFi-Sensing
This is a project about utilizing CSI data in wireless signals to apply machine learning methods in order to predict certain human activities in a given environment.

Note: This GitHub repo was originally a fork from [The ESP32 CSI Tool's repo](https://github.com/StevenMHernandez/ESP32-CSI-Tool). Hence, you can see there are other contributors since I want to keep credit to the authors of this fantastic toolkit.

![Alt text](/asset/architecture.png)

The project use two ESP32 microcontrollers as both TX and RX with the help of the ESP32 CSI tools for data collection as well as the Jetson Nano for on-edge model deployment.

All the logic for running on Jetson Nano is in the `read_csi.py`

Currently, we have trained 2 deep learing models:
- A CNN model which was trained on our dataset, is able to predict 3 activities with an accuracy of 0.95 and a loss of 0.02.
- A DNN model which was trained on another dataset, is able to predict 7 activities with an accuracy

This README will be updated with more details of the project in the future.

*To be continued...*