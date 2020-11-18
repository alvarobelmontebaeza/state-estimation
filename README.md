# STATE-ESTIMATION

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1) Clone the repository

git clone https://github.com/alvarobelmontebaeza/state-estimation

### 2) Build the package in your duckiebot

dts devel build -f -H HOSTNAME.local

### 3) Run the package specifying one of the provided launchers for the different tasks. In this case:

dts devel run -H HOSTNAME.local --launch <LAUNCHER>

The provided launchers are:
	- encoder_localization
	- at_localization
	- fused_localization

### 4) In another terminal, open RViz using the start_gui_tools utility

dts start_gui_tools HOSTNAME
rviz

Once opened, add TF and Camera image visualization

### 5) Open the virtual joystick or make the robot move autonomously and enjoy!

