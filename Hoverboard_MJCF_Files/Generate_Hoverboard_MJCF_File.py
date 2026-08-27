import os
import sys


#=================================================================================================================================================================================
os.system('cls' if os.name == 'nt' else 'clear')

#=================================================================================================================================================================================
# Hoverboard Dimensions
R = float(input("Enter Hoverboard Wheel Radius R [m]:"))
L = float(input("Enter the Hoverboard Wheelbase L [m]:"))
M = float(input("Enter Hoverboard Wheel Mass M [kg]:"))
h = float(input("Enter Inverted Pendulum Length h [m]:"))
m = float(input("Enter Inverted Pendulum Mass m [kg]:"))

#=================================================================================================================================================================================
# Generate MJCF File
def generate_MJCF(R,L,M,h,m):
    return f""" <mujoco model="hoverboard">
    <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>

    <default>
        <joint damping="0.01"/>
        <geom friction="1.0 0.005 0.0001"/>
    </default>

    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        
    </worldbody>
    """
