# ESP32-WiFi-Sensing
This is a project about utilizing CSI data in WiFi signals to apply machine learning methods in order to predict certain human activities in a given environment.

Note: This GitHub repo was originally a fork from [The ESP32 CSI Tool's repo](https://github.com/StevenMHernandez/ESP32-CSI-Tool). Hence, you can see there are other contributors since I want to keep credit to the authors of this fantastic toolkit.

![Alt text](/asset/architecture.png)

The project use two ESP32 microcontrollers as both TX and RX with the help of the ESP32 CSI tools for data collection as well as the Jetson Nano for on-edge model deployment. After running inference, all predictions will be published to an MQTT server and then the server will published these messages to all subcribers (including the Android applicaiton).

The dataset we collected is in the `datasets` folder. Although the CSI data collection has 7 activities, we only used 3 of them for training and evaluating machine learning models.

All the logic for running on Jetson Nano is in `python_utils/real_time_inference.py`

Currently, we have trained 4 machine learing models:
- A CNN model which has an accuracy of 0.95 and a loss of 0.02.
- A Random Forest model which has an accuracy of 0.93.
- A Linear Regression model which has an accuracy of 0.82.
- A Support Vector Machine model which has an accuracy of 0.83.

Model binary and notebook files can be found in their respective folders `models` and `notebooks`.