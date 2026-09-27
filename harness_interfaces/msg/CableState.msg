# Perceived wire centreline, from the anchor clamp to the connector's back face.
# In simulation these are the simulator's segment vertices plus perception noise.
std_msgs/Header header
geometry_msgs/Point[] points
float64 radius          # [m]
float64 length          # nominal wire length [m]
