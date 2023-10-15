#!/usr/bin/env python3

#   Copyright 2023 Grégori MIGNEROT, Élian BELMONTE, Benjamin STACH
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
    elif direction == '2':
        pub.publish(0b1000)
    elif direction == " ":
        pub.publish(0b10000)

if __name__ == '__main__':
    rospy.init_node('direction')
    pub = rospy.Publisher('/navigation/direction', UInt8, queue_size=1)
    while True:
        direction = input("Où aller ? Tout droit/Gauche/Droite :")
        if direction.upper() != "Q":
            chooseDirection(pub, direction)
        else:
            break
