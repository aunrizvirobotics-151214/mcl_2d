# mcl_2d
This project implements Monte Carlo Localization (MCL) using a particle filter approach for estimating the robot pose in a known map.

The system integrates:
- Probabilistic motion model (odometry-based)
- Observation model using sensor measurements
- Resampling strategies for particle filtering

The implementation is tested in Gazebo using the ArticubotOne robot.


## Simulation Setup
All algorithms are implemented and tested using:
- ROS2 (Humble)
- Gazebo
- ArticubotOne robot (based on John Evan’s tutorial)
