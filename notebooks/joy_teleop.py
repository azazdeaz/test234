# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import zmq
import numpy as np
import math
from IPython.display import clear_output, display
from inputs import get_gamepad
import msgpack
import time
import threading


context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.PUB)
socket.bind("tcp://0.0.0.0:5555")
# socket.connect("tcp://192.168.50.234:5555")

#  Do 10 requests, waiting each time for a response
# for request in range(10):
#     print("Sending request %s …" % request)
#     socket.send(b"Hello")

#     #  Get the reply.
#     message = socket.recv()
#     print("Received reply %s [ %s ]" % (request, message))


# %%
def joystickToDiff(x, y, minSpeed, maxSpeed, minJoystick=-1.0, maxJoystick=1.0):    # If x and y are 0, then there is not much to calculate...
    if x == 0.0 and y == 0.0:
        return (0.0, 0.0)
    
    direction = np.arctan2(y, x)
    xmax = np.minimum(1.0, np.abs(np.cos(direction)))
    ymax = np.minimum(1.0, np.abs(np.sin(direction)))
    speed = np.hypot(x,y) / np.hypot(xmax,ymax)
    display("d {}  s {} ".format(direction, speed))

    straight_treshold = np.pi / 8
    if np.abs(direction - np.pi/2.0) < straight_treshold:
        return (speed, speed)
    elif np.abs(direction + np.pi/2.0) < straight_treshold:
        return (-speed, -speed)
    # elif y > 0:

    # First Compute the angle in deg
    # First hypotenuse
    z = math.sqrt(x * x + y * y)

    # angle in radians
    rad = math.acos(math.fabs(x) / z)

    # and in degrees
    angle = rad * 180 / math.pi

    # Now angle indicates the measure of turn
    # Along a straight line, with an angle o, the turn co-efficient is same
    # this applies for angles between 0-90, with angle 0 the coeff is -1
    # with angle 45, the co-efficient is 0 and with angle 90, it is 1

    tcoeff = -1 + (angle / 90) * 2
    turn = tcoeff * math.fabs(math.fabs(y) - math.fabs(x))
    turn = round(turn * 100, 0) / 100

    # And max of y or x is the movement
    mov = max(math.fabs(y), math.fabs(x))

    # First and third quadrant
    if (x >= 0 and y >= 0) or (x < 0 and y < 0):
        rawLeft = mov
        rawRight = turn
    else:
        rawRight = mov
        rawLeft = turn

    # Reverse polarity
    if y < 0:
        rawLeft = 0 - rawLeft
        rawRight = 0 - rawRight

    # minJoystick, maxJoystick, minSpeed, maxSpeed
    # Map the values onto the defined rang
    rightOut = limit(rawRight, minJoystick, maxJoystick, minSpeed, maxSpeed)
    leftOut = limit(rawLeft, minJoystick, maxJoystick, minSpeed, maxSpeed)

    return (rightOut, leftOut)

def limit(v, in_min, in_max, out_min, out_max):
    # Check that the value is at least in_min
    if v < in_min:
        v = in_min
    # Check that the value is at most in_max
    if v > in_max:
        v = in_max
    return (v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

print(1,1,joystickToDiff(1,1,-100,100))
print(0,0,joystickToDiff(0,0,-100,100))
print(-1,-1,joystickToDiff(-1,-1,-100,100))


# %%
class Messages:
    def control(lr):
        return msgpack.packb({
            "type": "control-wheels",
            "left": lr[0],
            "right": lr[1]
        })

class Control():
    MODE_TRIGGER = "trigger"
    MODE_STICK = "stick"
    mode = 'trigger'

    rx = 0
    ry = 0
    trigger = [0.0, 0.0]
    bumper = [False, False]


    th_read_joy = None
    th_send_control = None
    
    @staticmethod
    def norm(value):
        low, high = 129.0, 32768.0
        return (np.maximum(0.0,np.abs(value)-low) / (high - low)) * np.sign(value)
    
    @classmethod
    def start(cls):
        cls.th_read_joy = threading.Thread(target=cls.read_joy)
        cls.th_read_joy.start()
        cls.th_send_control = threading.Thread(target=cls.send_control)
        cls.th_send_control.start()
        

    @classmethod
    def read_joy(cls):
        while True:
            events = get_gamepad()
            if len(events) == 0:
                print('sleep')
                time.sleep(0.1)
            else:
                for event in events:
                    # print(event.ev_type, event.code, event.state)
                    if event.ev_type == "Absolute":
                        if event.code == "ABS_RX":
                            cls.rx = cls.norm(event.state)
                        elif event.code == "ABS_RY":
                            cls.ry = -cls.norm(event.state)
                        elif event.code == "ABS_Z":
                            cls.trigger[0] = event.state / 255.0
                        elif event.code == "ABS_RZ":
                            cls.trigger[1] = event.state / 255.0
                    if event.ev_type == "Key":
                        if event.code == "BTN_TL":
                            cls.bumper[0] = event.state == 1
                        if event.code == "BTN_TR":
                            cls.bumper[1] = event.state == 1
                # print(cls.rx, cls.ry)
            # elif event.ev_type == "Sync" and event.code == "SYN_REPORT":
            #     self.x, self.y = 0, 0
    @classmethod
    def send_control(cls):
        while True:
            if cls.mode == cls.MODE_STICK:
                left, right = joystickToDiff(cls.rx, cls.ry, -1.0, 1.0)

            elif cls.mode == cls.MODE_TRIGGER:
                left = cls.trigger[0] 
                if cls.bumper[0]:
                    left *= -1
                right = cls.trigger[1]
                if cls.bumper[1]:
                    right *= -1

            # display((cls.rx, cls.ry))
            print(cls.mode, left, right)
            socket.send(Messages.control((left, right)))
            # socket.recv()
            time.sleep(0.1)


Control.start()
Control.th_read_joy.join()
# while True:
#     clear_output(wait=True)
#     c.read_joy()
#     c.send_controll()
#     time.sleep(0.01)


