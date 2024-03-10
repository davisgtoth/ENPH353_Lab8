
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(5)  # F,L,R,HL,HR
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)
        # cv2.waitKey(0)

        # NUM_BINS = 3
        NUM_BINS = 10
        state = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        # Convert to grayscale
        gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Binary threshold the image
        bin_thresh = 125
        ret, bin_frame = cv2.threshold(gray_frame, bin_thresh, 255, cv2.THRESH_BINARY)

        # find the road
        height, width  = bin_frame.shape
        buffer1 = 15
        buffer2 = 75
        radius = 10

        centreY1 = height - buffer1
        centreY2 = height - buffer2

        leftIndex1 = -1
        rightIndex1 = -1
        leftIndex2 = -1
        rightIndex2 = -1

        for i in range(width):
            if bin_frame[centreY1][i] == 0 and leftIndex1 == -1:
                leftIndex1 = i
            elif bin_frame[centreY1][i] == 0 and leftIndex1 != -1:
                rightIndex1 = i
            if bin_frame[centreY2][i] == 0 and leftIndex2 == -1:
                leftIndex2 = i
            elif bin_frame[centreY2][i] == 0 and leftIndex2 != -1:
                rightIndex2 = i
        
        roadCentreX1 = -1
        if leftIndex1 != -1 and rightIndex1 != -1:
            roadCentreX1 = (rightIndex1 + leftIndex1) // 2

        roadCentreX2 = -1
        if leftIndex2 != -1 and rightIndex2 != -1:
            roadCentreX2 = (rightIndex2 + leftIndex2) // 2

        # Draw circle on road and define state
        bin_divider = width / NUM_BINS
        if roadCentreX1 != -1:
            out_frame = cv2.circle(cv_image, (roadCentreX1, centreY1), radius, (0, 0, 255), -1)
            self.timeout = 0
            for i in range(NUM_BINS):
                if roadCentreX1 > bin_divider * i and roadCentreX1 < bin_divider * (i + 1):
                    state[0][i] = 1 
        else:
            out_frame = cv_image
            self.timeout += 1
        if roadCentreX2 != -1:
            # self.timeout = 0
            out_frame = cv2.circle(cv_image, (roadCentreX2, centreY2), radius, (0, 0, 255), -1)
            for i in range(NUM_BINS):
                if roadCentreX2 > bin_divider * i and roadCentreX2 < bin_divider * (i + 1):
                    state[1][i] = 1
        # if roadCentreX1 == -1 and roadCentreX2 == -1:
        #     out_frame = cv_image
        #     self.timeout += 1

        # Check if the line has been lost for too long
        if self.timeout > 20:
            done = True
        
        ## draw rectangular dividers on output frame
        # for i in range(NUM_BINS):
        #     out_frame = cv2.line(out_frame, (int(bin_divider * (i + 1)), 0), (int(bin_divider * (i + 1)), height), (0, 0, 0), 2)
        
        # draw state on output frame
        cv2.putText(out_frame, str(state[1]), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(out_frame, str(state[0]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("output", out_frame)
        cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5
        elif action == 3: # HARD LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 1.0
        elif action == 4: # HARD RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -1.0

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 3
                # print("Action: Forward, Reward: {}".format(reward))
            elif action == 1:  # LEFT
                reward = 2
                # print("Action: Left, Reward: {}".format(reward))
            elif action == 2: # RIGHT
                reward = 2  
                # print("Action: Right, Reward: {}".format(reward))
            elif action == 3: # HARD LEFT
                reward = 1 
                # print("Action: Hard Left, Reward: {}".format(reward))
            elif action == 4: # HARD RIGHT
                reward = 1 
                # print("Action: Hard Right, Reward: {}".format(reward))
        else:
            reward = -200

        # print("Action: {}, Reward: {}".format(action, reward))

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
