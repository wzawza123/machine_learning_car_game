'''
Description: this program implement a environment for the car game.
Date: 2022-06-10 15:50:52
LastEditTime: 2022-06-13 17:14:16
'''

# import necessary libraries
from asyncio import to_thread
from time import sleep
from tkinter.messagebox import NO
from matplotlib import projections
import pygame
import sys
import numpy as np
import math
import random

# import modules
from car import *
from common import *
from map_generator import *


class Environment:
    """
    本类为赛车环境类，用以实现赛车环境的操作
    """
    def __init__(self) -> None:
        """
        本函数用于环境类的初始化
        """
        self.screen = None
        # initialize the car_list
        self.car_list = []
        # initialize the pygame
        pygame.init()

    def start(self, return_mode="distances", visible=True, max_frame=20000,seed=None):
        """本函数用以启动环境，在进行实例化后调用一次本函数即可，此后可通过reset函数来进行环境重置

        Args:
            return_mode (str, optional): 环境交互返回信息方式，"distances"返回5方向的距离，"frames"返回当前画面的颜色矩阵(720,1080,3),"direction and distances"返回当前速度以及速度在正确方向上的分量，同时返回5方向距离. Defaults to "distances".
            visible (bool, optional): 是否将环境运行过程显示在屏幕上. Defaults to True.
            max_frame (int, optional): 环境运行的最大帧数，若达到该帧数agent尚未停止运行则立即停止. Defaults to 20000.
            seed (int, optional): 作为环境生成时的随机种子，若为None则随机. Defaults to None.
        
        Raises:
            ValueError: 当return_mode不为"distances"或"frames"或"direction and distances"时，抛出该异常
        """
        # copy parameters
        if return_mode == "distances":
            self.return_mode = "distances"
        elif return_mode == "frames":
            self.return_mode = "frames"
        elif return_mode == "direction and distances":
            self.return_mode = "direction and distances"
        else:
            raise ValueError("return_mode must be 'distances' or 'frames'")
        self.return_mode = return_mode
        self.visiable = visible
        self.max_frame = max_frame
        self.clock = pygame.time.Clock()
        
        # generate the map
        generate_map(seed)
        
        self.reset()

    def reset(self):
        """本函数用以重置环境，此后可以直接调用step进行交互
        """
        # load the map
        # initialize the pygame
        if not self.visiable:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT),flags=pygame.HIDDEN)
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.car_image = pygame.image.load('car.png').convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (CAR_WIDTH, CAR_HEIGHT))
        self.background_img, self.start_point = load_map('./maps/', 'map')
        #draw the map
        self.screen.blit(self.background_img, (0, 0))
        pygame.display.flip()

        self.car_list = []
        self.car_list.append(Car(self.car_image, self.start_point, np.pi / 2, True))
        self.running = True
        self.cur_step=0
        
        

    def step(self, action:List[float]):
        """该函数实现环境交互，可以调用reset函数来重置环境

        Args:
            action (List[float,float,float,float]): 输入本帧agent进行的动作，第一个元素为向前加速度[0,0.2]，第二个元素为向后加速度[0,0.2]，第三个元素为左转角速度[0,0.03]，第四个元素为右转角速度[0,0.03]

        Returns:
            for "distances" return_mode:
                float: speed: 车辆的运行速度
                list[float]: state: 车辆5个方向离墙体的的距离
            for "frames" return_mode:
                list[list[list[int]]]: color array: with the shape of (1080,720,3): 帧颜色矩阵, 范围[0,255]
            float: reward: 当前行动的奖励
            bool: done: 是否停止运行
            str: info: 当前状态的信息
                "already stopped" : 车辆已经停止运行
                "lap completed" : 车辆已经完成一圈
                "reach max_frame" : 车辆已经达到最大帧数
                "collision" : 车辆与障碍物发生碰撞
        """
        if not self.running:
            return None, None, True, "already stopped"
        #update the ctr values
        self.cur_step+=1
        info=""
        #update the car
        for car in self.car_list:
            state_dist,cur_d_angle,cur_angle=car.update_by_step(self.screen,action)
            speed_vec=car.get_speed_vec()
            correct_vec=calculate_correct_vec(540,360,car.x,car.y)
            projection=calculate_projection(speed_vec,correct_vec)
            # print(speed_vec,correct_vec,projection)
            cur_reward=REWARD_ONE_LAP/(2*np.pi)*cur_d_angle
            # test whether the agent finish the game
            if cur_angle>2*np.pi:
                self.running=False
                info="lap completed"

        # test if the frame reach the max_frame
        if self.cur_step>=self.max_frame:
                self.running=False
                info="reach max_frame"

        # draw the whole map
        self.screen.blit(self.background_img, (0, 0))
        # draw the car
        self.screen.blit(self.background_img, (0, 0))
        for car in self.car_list:
            car.detect_track_boundary(
                self.screen)  # for the function is using the pixel value to detect the track boundary, so it should be done before the car.draw()
            # test the car is alive
            if not car.is_alive:
                self.running = False
                info="collision"
                cur_reward+=DEAD_SCORE

            car.calculate_distance(self.screen)  # for the same reson, it should be done before the car.draw()
            car.draw(self.screen)
        # draw a circle
        # pygame.draw.circle(self.screen, (255, 0, 0), (int(510), int(360)), 10)
        pygame.display.flip()
        if self.return_mode == "distances":
            return self.car_list[0].speed,state_dist,cur_reward,not(self.running),info
        else:
            return self.get_cur_frame(),cur_reward,not(self.running),info
    
    def get_cur_frame(self):
        """该函数用以获取当前帧的颜色矩阵
        """
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def get_cur_step(self):
        """该函数用以获取当前步数
        """
        return self.cur_step


    def close(self):
        """调用该函数用以关闭环境
        """
        pygame.quit()
    
    def get_position(self):
        """获取当前车辆的位置

        Returns:
            (int,int): x,y车辆坐标
        """
        return self.car_list[0].x,self.car_list[0].y

def load_map_test():
    """本函数为测试函数，用以测试load_map函数
    """
    pygame.init()
    pygame.display.set_caption('Car')
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    car_image = pygame.image.load('car.png').convert_alpha()
    car_image = pygame.transform.scale(car_image, (CAR_WIDTH, CAR_HEIGHT))
    background_img, start_point = load_map('./maps/', 'map')
    screen.blit(background_img, (0, 0))
    pygame.display.flip()

def env_test():
    """本函数为调用环境的样例程序
    """
    env = Environment()
    env.start("distances",True,12000,120)
    action = [ACCELERATION_FORWARD,0,0,0]
    score=0
    while(True):
        speed,state,reward,done,info=env.step(action)
        score+=reward
        action=simple_auto(speed,state)
        # print(state,reward,done,info)
        # print(env.get_position())
        if done:
            break
        sleep(0.01)
    print("tot score:",score)
    print("tot step:",env.get_cur_step())
    env.close()

if __name__ == "__main__":
    env_test()
