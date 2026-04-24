from physdnet.xtf_utils import *

# Load XTF 
xtf_file_path = "data/2025-09-24_09-25-24_0.xtf"
xtf_pings = load_xtf(xtf_file_path)

# Extract data 
swaths, trajectory, altitude, roll, pitch, yaw = calculate_swath_positions(xtf_pings)
waterfall_img, sonar_img = calculate_waterfall(xtf_pings)
port_idx, stbd_idx = calculate_blind_zone_indices(xtf_pings)

import matplotlib.pyplot as plt 

_, ax = plt.subplots(figsize=(10,5))

# Plot altitude
ax.plot(altitude)
ax.set_title("Altitude")
ax.set_xlabel("Ping Number [#]")
ax.set_ylabel("Altitude [metres]")

plt.show()

from math import pi 

# Plot orientations
_, ax = plt.subplots(figsize=(10,5))
ax.plot(roll*180/pi)
ax.set_title("Roll")
ax.set_xlabel("Ping Number [#]")
ax.set_ylabel("Rotation [deg]")

_, ax = plt.subplots(figsize=(10,5))
ax.plot(pitch*180/pi)
ax.set_title("Pitch")
ax.set_xlabel("Ping Number [#]")
ax.set_ylabel("Rotation [deg]")

_, ax = plt.subplots(figsize=(10,5))
ax.plot(yaw*180/pi)
ax.set_title("Yaw")
ax.set_xlabel("Ping Number [#]")
ax.set_ylabel("Rotation [deg]")

plt.show()


_, ax = plt.subplots(figsize=(10,5))

# Plot trajectory
ax.plot(trajectory[:,0], trajectory[:,1], color="blue", label="trajectory")

# Plot ping lines 
step = 50
for i in range(0, swaths.shape[0], step):
    start_point = swaths[i, 0, :]      
    end_point = swaths[i, -1, :]       
    
    x_vals = [start_point[0], end_point[0]]
    y_vals = [start_point[1], end_point[1]]
    
    if i == 0:
        ax.plot(x_vals, y_vals, color='green', linestyle='-', alpha=0.4, linewidth=1, label=f"swaths with step={step}")
    else:    
        ax.plot(x_vals, y_vals, color='green', linestyle='-', alpha=0.4, linewidth=1)

ax.axis("equal")
ax.set_title("Trajectory")
ax.set_xlabel("Easting [metres]")
ax.set_ylabel("Northing [metres]")
ax.axis("equal")
ax.scatter([trajectory[0,0]], [trajectory[0,1]], label="Start", c="red")
ax.scatter([trajectory[-1,0]], [trajectory[-1,1]], label="End", c="green")
ax.legend()

plt.show()

N = waterfall_img.shape[0]
ping_idx = slice(0, N)

# Plot original and processed images
_, ax1 = plt.subplots(figsize=(40,20))
ax1.imshow(sonar_img[ping_idx,:] , cmap="gray")
ax1.set_title("Original Waterfall")

_, ax2 = plt.subplots(figsize=(40,20))
ax2.imshow(waterfall_img[ping_idx,:] , cmap="gray")
ax2.set_title("Processed Waterfall")

plt.show()