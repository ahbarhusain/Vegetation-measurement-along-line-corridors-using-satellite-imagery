# DRUMA 

This is repository about my university final year project.

## Description

Druma is an Artifical Intelligence System project made by students of Bandung Institute of Technology. Druma is made to detect trees that have the potential to harm the power line. Druma uses Deepforest library that has been trained with more than 20 thousand trees to detect trees, and Hough Transform algorithm to detect power lines. Druma can be used to predict your images from UAVs or satellite. The detected tree in the image will be marked with differetn box color based on their distance from the nearest power line. The color indicates wether the trees is growing inside the ROW (Right of Way) of power line or not.

## Getting Started

### System Design

Druma has 2 sub-system:
* Web Client / User Interface
    - Web Client act as an user interface of the system. User can input the image from the web client and the web client will show the result.
* Backend
    - Deepforest API: This API will run a deep learning model from "Deepforest" for detecting the tree crown in the image.
    - Hough Transform API: This API will run Hough Transform for detecting power line in the image.

### Requirement

Each sub-system will be deployed in different VM with following specification:
* Web Client / User Interface
    ```
    OS Ubuntu Server 18.04
    vCPU: 1 core
    Memory: 3.75 GB 
    Storage: 25 GB 
    ```
* Backend
    - Deepforest API:
    ```
    OS Ubuntu Server 18.04
    vCPU: 2 core
    Memory: 7.5 GB 
    Storage: 30 GB 
    ```
    - Hough Transform API:
    ```
    OS Ubuntu Server 18.04
    vCPU: 2 core
    Memory: 7.5 GB 
    Storage: 25 GB 
    ```

## Authors

* [Gelar Pambudi Adhiluhung](https://github.com/gelarpambudi) 
* [Karisa Ardelia Hanifah](https://github.com/karisaardelia)

## Acknowledgments

* [Deepforest](https://github.com/weecology/DeepForest)
* [PCNN Filter](https://github.com/arnoldcvl/modified-pcnn)
* [Towards automatic power line detection for a UAV surveillance system using pulse coupled neural filter and an improved Hough transform](https://www.researchgate.net/publication/40220506_Towards_automatic_power_line_detection_for_a_UAV_surveillance_system_using_pulse_coupled_neural_filter_and_an_improved_Hough_transform)
* [Advances in vegetation management for power line corridor monitoring using aerial remote sensing techniques](https://ieeexplore.ieee.org/document/5624431)
