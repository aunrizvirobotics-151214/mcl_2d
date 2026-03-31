# mcl_2d
This project implements Monte Carlo Localization (MCL) using a particle filter approach for estimating the robot pose in a known map.

The system integrates:
- Odometry motion model
- Observation model using sensor measurements
- Resampling strategies for particle filtering

The implementation is tested in Gazebo using the ArticubotOne robot.

<img width="1920" height="1080" alt="Screenshot from 2026-03-31 17-33-07" src="https://github.com/user-attachments/assets/638a60e2-d8a4-4f87-a551-52b66dd47e7d" />

<img width="1920" height="1080" alt="Screenshot from 2026-03-31 17-32-59" src="https://github.com/user-attachments/assets/ead34c48-3662-4c79-a19a-3f3d75dfa795" />

<img width="1920" height="1080" alt="Screenshot from 2026-03-31 17-32-56" src="https://github.com/user-attachments/assets/7f43122f-4c48-4eea-aafd-d16fdc239764" />

<img width="1920" height="1080" alt="Screenshot from 2026-03-31 17-30-57" src="https://github.com/user-attachments/assets/63be3a26-cf57-4544-9378-e1e1da1cee6b" />

## Simulation Setup
All algorithms are implemented and tested using:
- ROS2 (Humble)
- Gazebo
- ArticubotOne robot (based on John Evan’s tutorial)
