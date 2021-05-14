# Interceptor 4.0 - Team Delamain

![Python version](https://img.shields.io/badge/python-v3.7-blue)
[![GitHub license](https://img.shields.io/github/license/heng2j/delamain)](https://github.com/heng2j/delamain/master/LICENSE)


Project for DGMD E-17, Harvard University, Spring 2021

Team Members:
   * Zhongheng (Heng) Li aka heng2j
   * Nicolas Morant aka DevGlitch
   * Huayu (Jack) Tsu aka codejacktsu

<br>

<br>

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/heng2j/delamain">
    <img src="images/police_car_picture.png" alt="Logo" height="300">
  </a>
</p>

<!-- DESCRIPTION OF THE PROJECT -->
## Description


<!-- PROJECT REPORT-->
## Project Report
https://github.com/heng2j/delamain/Presentation_and_Report/Project_Report_Team_Delamain.pdf


<!-- PROJECT PRESENTATION-->
## Presentation
<p align="center">
  <a href="https://youtu.be/7PMrhMN3heU">
    <img src="images/youtube.jpeg" alt="Logo" width="200" height="200">
  </a>
</p>


<!-- DEMO OF THE PROJECT -->
## Demos
https://github.com/heng2j/delamain/Demos


<!-- GETTING STARTED -->
## Getting Started

Follow the below instructions in order to run Interceptor 4.0 on your machine.


### Prerequisites

* CARLA 0.9.10 --> https://github.com/carla-simulator/carla

This project was developed using this specific version of CARLA.
<br>
We cannot guarantee that it would work with any higher version.


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/heng2j/delamain.git
   ```
2. Install any missing packages. 
   + We recommend using a conda environment with Python 3.7.
    

### Running

1. Start Server Map: Open CARLAUE4.exe


2. Load the map that you'd lile. By default CARLA loads map Town03


3. Run any of the command below.<br>
   Please make sure you are in the correct directory.
   

* #### Base Model 3

Third iteration of our base model with full capabilities.<br>
Press "i" to activate GPS from current location to destination.<br>
Press "p" to toggle autopilot<br>
Press "h" to toggle hotkey map
```sh
   python base_model3.py
   ```

* #### GPS Navigation Base Model

This file gives you a short demo of the navigation system. You can change the destination location.
Please ensure the destination is for the load CARLA map.
```sh
   python base_model_nav.py
   ```

* #### Self-Parking Base Model

This file gives you a short demo of the self-parking feature. In this file you have the option to change from perpendicular to parallel parking.
```sh
   python base_model_park.py
   ```

* #### Road Network Map (gps-nav directory)

This script enables you to visualize the road network of any CARLA map.
```sh
   python road_network_map.py
   ```

* #### Spectator Location (gps-nav directory)

This script gave you the ability to get the exact location of the spectator view. 
It gives you the location in CARLA Location (x, y, z).
```sh
   python spectator_location.py
   ```

* #### Topology Edge & Node (gps-nav directory)

This script enables you to store the topology data (edges and nodes) in two parquet files.
These files are in a format that enables you to use with Network X.
```sh
   python topology_edge_and_node.py.py
   ```

* #### Topology Waypoints Visualizer (gps-nav directory)

This script is a visualizer in CARLA of the topology waypoints. Make sure to run the previous script in order for this one to work.
```sh
   python topology_waypoints_visualizer.py.py
   ```

* #### ... (... directory)
```sh
   python FILENAME.py
   ```

* #### ... (... directory)
```sh
   python FILENAME.py
   ```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Zhongheng (Heng) Li  - [Github](https://github.com/heng2j)

Nicolas Morant - [Personal Website](https://www.nicolasmorant.com/)
 & [Github](https://github.com/DevGlitch)

Huayu (Jack) Tsu - [Github](https://github.com/codejacktsu)

<br>

Project Link: [https://github.com/heng2j/delamain](https://github.com/heng2j/delamain)
