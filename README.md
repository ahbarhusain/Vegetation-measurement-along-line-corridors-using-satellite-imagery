# NebulaQuest


## Description

NebulaQuest is an Artifical Intelligence System.NebulaQuest is made to detect trees that have the potential to harm the power line. NebulaQuest uses Deepforest library that has been trained with more than 20 thousand trees to detect trees, and Hough Transform algorithm to detect power lines. NebulaQuest can be used to predict your images from UAVs or satellite. The detected tree in the image will be marked with differetn box color based on their distance from the nearest power line. The color indicates wether the trees is growing inside the ROW (Right of Way) of power line or not.

## Getting Started

### System Design

NebulaQuest has 2 sub-system:
* Web Client / User Interface
    - Web Client act as an user interface of the system. User can input the image from the web client and the web client will show the result.
* Backend
    - Deepforest API: This API will run a deep learning model from "Deepforest" for detecting the tree crown in the image.
    - Hough Transform API: This API will run Hough Transform for detecting power line in the image.

![nebula](https://github.com/ahbarhusain/Vegetation-measurement-along-line-corridors-using-satellite-imagery/assets/97535337/de87b853-cdfb-495d-9f19-72f5a9ad9d2a)

