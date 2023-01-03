#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import UInt8

def chooseDirection(pub, direction):
    if direction == '\x1b[A':
        pub.publish(0b0001)
    elif direction == '\x1b[D':
        pub.publish(0b0010)
    elif direction == '\x1b[C':
        pub.publish(0b0100)

def getInput():
    while True:
        direction = input("Où aller ? Tout droit/Gauche/Droite :")
        if direction == '\x1b[A' or direction == '\x1b[C' or direction == '\x1b[D':
            return direction
        elif direction.upper() == 'Q':
            return None
        else:
            print('Erreur ! Veuillez utiliser les touches directionnelles')
    

if __name__ == '__main__':
    rospy.init_node('direction')
    pub = rospy.Publisher('/navigation/direction', UInt8, queue_size=1)
    while True:
        direction = getInput()
        if direction is not None:
            chooseDirection(pub, direction)
        else:
            break
