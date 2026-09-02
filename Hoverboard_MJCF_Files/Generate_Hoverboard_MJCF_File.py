import os
import sys
import tkinter as tk
from tkinter import filedialog

#=================================================================================================================================================================================
os.system('cls' if os.name == 'nt' else 'clear')

#=================================================================================================================================================================================
# Hoverboard Dimensions
R = float(input("Enter Hoverboard Wheel Radius R [m]:"))
L = float(input("Enter the Hoverboard Wheelbase L [m]:"))
M = float(input("Enter Hoverboard Wheel Mass M [kg]:"))
h = float(input("Enter Inverted Pendulum Length h [m]:"))
m = float(input("Enter Inverted Pendulum Mass m [kg]:"))
T_max = float(input("Enter Maximum Motor Torque Output T_max [N.m]:"))
filename = input("Enter the Filename:")

#=================================================================================================================================================================================
# Define MJCF File
def generate_MJCF(R,L,M,h,m,T_max):
    return f""" <mujoco model="hoverboard">
    <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>

    <default>
        <joint damping="0.01"/>
        <geom friction="1.0 0.005 0.0001"/>
    </default>

    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <body name="chassis" pos="0 0 {R}">
            <joint type="free"/>
            <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-8 1e-8 1e-8"/>
            <geom name="chassis_geom" type="cylinder" size="0.02 {L/2}" quat="0.707 0.707 0 0" rgba="0.6 0.6 0.6 1"/>

            <body name="left_wheel" pos="0 {L/2} 0">
                <joint name="left_hinge" type="hinge" axis="0 1 0" damping="0.001"/>
                <inertial pos="0 0 0" mass="{M}" diaginertia="{(M*R**2)/4} {(M*R**2)/2} {(M*R**2)/4}"/>
                <geom name="left_wheel_geom" type="cylinder" size="{R} 0.02" quat="0.707 0.707 0 0" rgba="0 0 1 1"/>
            </body>

            <body name="right_wheel" pos="0 {-L/2} 0">
                <joint name="right_hinge" type="hinge" axis="0 1 0" damping="0.001"/>
                <inertial pos="0 0 0" mass="{M}" diaginertia="{(M*R**2)/4} {(M*R**2)/2} {(M*R**2)/4}"/>
                <geom name="right_wheel_geom" type="cylinder" size="{R} 0.02" quat="0.707 0.707 0 0" rgba="1 0 0 1"/>
            </body>

            <body name="pendulum_gimble" pos="0 0 0">
                <joint name = "pend_hinge_x" type="hinge" axis="1 0 0" damping="0.0005"/>
                <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-8 1e-8 1e-8"/>

                <body name="pendulum_rod" pos="0 0 0">
                    <joint name="pend_hinge_y" type="hinge" axis="0 1 0" damping="0.0005"/>
                    <inertial pos="0 0 {h/2}" mass="1e-6" diaginertia="1e-8 1e-8 1e-8"/>
                    <geom name="rod_geom" type="capsule" fromto="0 0 0 0 0 {h}" size="0.015" rgba="0.6 0.6 0.6 1"/>

                    <body name="pendulum_mass" pos="0 0 {h}">
                        <inertial pos="0 0 0" mass="{m}" diaginertia="0.001 0.001 0.001"/>
                        <geom name="mass_geom" type="sphere" size="0.04" rgba="0 1 0 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="left_motor" joint="left_hinge" gear="1" ctrlrange="{-T_max} {T_max}"/>
        <motor name="right_motor" joint="right_hinge" gear="1" ctrlrange="{-T_max} {T_max}"/>
    </actuator>

    </mujoco>"""

#=================================================================================================================================================================================
# Generate and Save MJCF File
hoverboard = generate_MJCF(R,L,M,h,m,T_max)
path = "/mnt/c/Users/admin/Documents/GitHub/Hoverboard_RL_Controls/Hoverboard_MJCF_Files"
if not filename.endswith(".xml"):
    filename +=".xml"

full_path = os.path.join(path, filename)
with open(full_path,"w", encoding="utf-8") as f:
    f.write(hoverboard)
print(f"Saved: {full_path}")

