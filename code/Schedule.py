import tensorflow as tf
import numpy as np
import os
from MADDPG37 import MADDPG_FL
from FL_Information11 import FL_Agent
from MEMORY import Memory
import matplotlib
import random
# matplotlib.use('TkAgg')
import time
import heapq
import math
import copy
from matplotlib.font_manager import *
from Differential_Evolution import Fitness
from Differential_Evolution import DE
from Differential_Evolution import selection


from utils_test.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils_test.options import args_parser
from models_test.Update import LocalUpdate
from models_test.Nets import MLP, CNNMnist, CNNCifar
from models_test.Fed import FedAvg
from models_test.test import test_img
from local_update import local_one
from local_update import caculate
from local_update import initial

start = time.clock()
np.random.seed(1)
tf.set_random_seed(1)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#参数设置
MAX_EPISODES = 30
MAX_EP_STEPS = 10
MEMORY_CAPACITY = 100000
BATCH_SIZE = 256
gamma = 0.95
noise = 1e-12
popsize =10
epsilon = 0.1
T_meta = 10

def test():
    L = list(range(N))
    return L

def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]
    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                     zip(online_var, target_var)]  # target network 的更新
    return target_init, target_update

def Pop_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]
    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, online + target) for online, target in
                     zip(online_var, target_var)]  # target network 的更新
    return target_init,target_update

def Meta_update(oneline_name, target_name, popsize, epsilon):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]
    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, target + epsilon * (target * popsize - online)) for online, target in zip(online_var, target_var)]
    return target_init, target_update

def Initial_update(oneline_name, target_name):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]
    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    return target_init

def necessary(b_M):
    f_state = b_M[:, 0:550]
    f1_b_action = b_M[:, 550:551]
    f2_b_action = b_M[:, 551:552]
    f3_b_action = b_M[:, 552:553]
    f4_b_action = b_M[:, 553:554]
    f5_b_action = b_M[:, 554:555]
    f6_b_action = b_M[:, 555:556]
    f7_b_action = b_M[:, 556:557]
    f8_b_action = b_M[:, 557:558]
    f9_b_action = b_M[:, 558:559]
    f10_b_action = b_M[:, 559:560]
    f1_b_reward = b_M[:, 560:561]
    f2_b_reward = b_M[:, 561:562]
    f3_b_reward = b_M[:, 562:563]
    f4_b_reward = b_M[:, 563:564]
    f5_b_reward = b_M[:, 564:565]
    f6_b_reward = b_M[:, 565:566]
    f7_b_reward = b_M[:, 566:567]
    f8_b_reward = b_M[:, 567:568]
    f9_b_reward = b_M[:, 568:569]
    f10_b_reward = b_M[:, 569:570]
    b_state_next = b_M[:, 570:1120]
    f1_b_action_next = agent_1_target.next_action(b_state_next[:, 0:55], sess)
    f2_b_action_next = agent_2_target.next_action(b_state_next[:, 55:110], sess)
    f3_b_action_next = agent_3_target.next_action(b_state_next[:, 110:165], sess)
    f4_b_action_next = agent_4_target.next_action(b_state_next[:, 165:220], sess)
    f5_b_action_next = agent_5_target.next_action(b_state_next[:, 220:275], sess)
    f6_b_action_next = agent_6_target.next_action(b_state_next[:, 275:330], sess)
    f7_b_action_next = agent_7_target.next_action(b_state_next[:, 330:385], sess)
    f8_b_action_next = agent_8_target.next_action(b_state_next[:, 385:440], sess)
    f9_b_action_next = agent_9_target.next_action(b_state_next[:, 440:495], sess)
    f10_b_action_next = agent_10_target.next_action(b_state_next[:, 495:550], sess)
    return f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next

agent_1 = MADDPG_FL('agent_1')
agent_1_target = MADDPG_FL('agent_1_target')
agent_2 = MADDPG_FL('agent_2')
agent_2_target = MADDPG_FL('agent_2_target')
agent_3 = MADDPG_FL('agent_3')
agent_3_target = MADDPG_FL('agent_3_target')
agent_4 = MADDPG_FL('agent_4')
agent_4_target = MADDPG_FL('agent_4_target')
agent_5 = MADDPG_FL('agent_5')
agent_5_target = MADDPG_FL('agent_5_target')
agent_6 = MADDPG_FL('agent_6')
agent_6_target = MADDPG_FL('agent_6_target')
agent_7 = MADDPG_FL('agent_7')
agent_7_target = MADDPG_FL('agent_7_target')
agent_8 = MADDPG_FL('agent_8')
agent_8_target = MADDPG_FL('agent_8_target')
agent_9 = MADDPG_FL('agent_9')
agent_9_target = MADDPG_FL('agent_9_target')
agent_10 = MADDPG_FL('agent_10')
agent_10_target = MADDPG_FL('agent_10_target')
meta_1_target = MADDPG_FL('meta_1_target')
meta_2_target = MADDPG_FL('meta_2_target')
meta_3_target = MADDPG_FL('meta_3_target')
meta_4_target = MADDPG_FL('meta_4_target')
meta_5_target = MADDPG_FL('meta_5_target')
meta_6_target = MADDPG_FL('meta_6_target')
meta_7_target = MADDPG_FL('meta_7_target')
meta_8_target = MADDPG_FL('meta_8_target')
meta_9_target = MADDPG_FL('meta_9_target')
meta_10_target = MADDPG_FL('meta_10_target')

agent_1_actor_target_init, agent_1_actor_target_update = create_init_update('agent_1_actor', 'agent_1_target_actor')
agent_1_critic_target_init, agent_1_critic_target_update = create_init_update('agent_1_critic', 'agent_1_target_critic')
agent_2_actor_target_init, agent_2_actor_target_update = create_init_update('agent_2_actor', 'agent_2_target_actor')
agent_2_critic_target_init, agent_2_critic_target_update = create_init_update('agent_2_critic', 'agent_2_target_critic')
agent_3_actor_target_init, agent_3_actor_target_update = create_init_update('agent_3_actor', 'agent_3_target_actor')
agent_3_critic_target_init, agent_3_critic_target_update = create_init_update('agent_3_critic', 'agent_3_target_critic')
agent_4_actor_target_init, agent_4_actor_target_update = create_init_update('agent_4_actor', 'agent_4_target_actor')
agent_4_critic_target_init, agent_4_critic_target_update = create_init_update('agent_4_critic', 'agent_4_target_critic')
agent_5_actor_target_init, agent_5_actor_target_update = create_init_update('agent_5_actor', 'agent_5_target_actor')
agent_5_critic_target_init, agent_5_critic_target_update = create_init_update('agent_5_critic', 'agent_5_target_critic')
agent_6_actor_target_init, agent_6_actor_target_update = create_init_update('agent_6_actor', 'agent_6_target_actor')
agent_6_critic_target_init, agent_6_critic_target_update = create_init_update('agent_6_critic', 'agent_6_target_critic')
agent_7_actor_target_init, agent_7_actor_target_update = create_init_update('agent_7_actor', 'agent_7_target_actor')
agent_7_critic_target_init, agent_7_critic_target_update = create_init_update('agent_7_critic', 'agent_7_target_critic')
agent_8_actor_target_init, agent_8_actor_target_update = create_init_update('agent_8_actor', 'agent_8_target_actor')
agent_8_critic_target_init, agent_8_critic_target_update = create_init_update('agent_8_critic', 'agent_8_target_critic')
agent_9_actor_target_init, agent_9_actor_target_update = create_init_update('agent_9_actor', 'agent_9_target_actor')
agent_9_critic_target_init, agent_9_critic_target_update = create_init_update('agent_9_critic', 'agent_9_target_critic')
agent_10_actor_target_init, agent_10_actor_target_update = create_init_update('agent_10_actor', 'agent_10_target_actor')
agent_10_critic_target_init, agent_10_critic_target_update = create_init_update('agent_10_critic',
                                                                                'agent_10_target_critic')
pop_1 = MADDPG_FL('pop_1')
pop_2 = MADDPG_FL('pop_2')
pop_3 = MADDPG_FL('pop_3')
pop_4 = MADDPG_FL('pop_4')
pop_5 = MADDPG_FL('pop_5')
pop_6 = MADDPG_FL('pop_6')
pop_7 = MADDPG_FL('pop_7')
pop_8 = MADDPG_FL('pop_8')
pop_9 = MADDPG_FL('pop_9')
pop_10 = MADDPG_FL('pop_10')

meta_1 = MADDPG_FL('meta_1')
meta_2 = MADDPG_FL('meta_2')
meta_3 = MADDPG_FL('meta_3')
meta_4 = MADDPG_FL('meta_4')
meta_5 = MADDPG_FL('meta_5')
meta_6 = MADDPG_FL('meta_6')
meta_7 = MADDPG_FL('meta_7')
meta_8 = MADDPG_FL('meta_8')
meta_9 = MADDPG_FL('meta_9')
meta_10 = MADDPG_FL('meta_10')


pop_1_actor_init, pop_1_actor_update = Pop_update('agent_1_actor','pop_1_actor')
pop_1_critic_init, pop_1_critic_update = Pop_update('agent_1_critic','pop_1_critic')
pop_2_actor_init, pop_2_actor_update = Pop_update('agent_2_actor','pop_2_actor')
pop_2_critic_init, pop_2_critic_update = Pop_update('agent_2_critic','pop_2_critic')
pop_3_actor_init, pop_3_actor_update = Pop_update('agent_3_actor','pop_3_actor')
pop_3_critic_init, pop_3_critic_update = Pop_update('agent_3_critic','pop_3_critic')
pop_4_actor_init, pop_4_actor_update = Pop_update('agent_4_actor','pop_4_actor')
pop_4_critic_init, pop_4_critic_update = Pop_update('agent_4_critic','pop_4_critic')
pop_5_actor_init, pop_5_actor_update = Pop_update('agent_5_actor','pop_5_actor')
pop_5_critic_init, pop_5_critic_update = Pop_update('agent_5_critic','pop_5_critic')
pop_6_actor_init, pop_6_actor_update = Pop_update('agent_6_actor','pop_6_actor')
pop_6_critic_init, pop_6_critic_update = Pop_update('agent_6_critic','pop_6_critic')
pop_7_actor_init, pop_7_actor_update = Pop_update('agent_7_actor','pop_7_actor')
pop_7_critic_init, pop_7_critic_update = Pop_update('agent_7_critic','pop_7_critic')
pop_8_actor_init, pop_8_actor_update = Pop_update('agent_8_actor','pop_8_actor')
pop_8_critic_init, pop_8_critic_update = Pop_update('agent_8_critic','pop_8_critic')
pop_9_actor_init, pop_9_actor_update = Pop_update('agent_9_actor','pop_9_actor')
pop_9_critic_init, pop_9_critic_update = Pop_update('agent_9_critic','pop_9_critic')
pop_10_actor_init, pop_10_actor_update = Pop_update('agent_10_actor','pop_10_actor')
pop_10_critic_init, pop_10_critic_update = Pop_update('agent_10_critic','pop_10_critic')

meta_1_actor_init, meta_1_actor_update = Meta_update('pop_1_actor', 'meta_1_actor', popsize, epsilon)
meta_1_critic_init, meta_1_critic_update = Meta_update('pop_1_critic', 'meta_1_critic', popsize, epsilon)
meta_2_actor_init, meta_2_actor_update = Meta_update('pop_2_actor', 'meta_2_actor', popsize, epsilon)
meta_2_critic_init, meta_2_critic_update = Meta_update('pop_2_critic', 'meta_2_critic', popsize, epsilon)
meta_3_actor_init, meta_3_actor_update = Meta_update('pop_3_actor', 'meta_3_actor', popsize, epsilon)
meta_3_critic_init, meta_3_critic_update = Meta_update('pop_3_critic', 'meta_3_critic', popsize, epsilon)
meta_4_actor_init, meta_4_actor_update = Meta_update('pop_4_actor', 'meta_4_actor', popsize, epsilon)
meta_4_critic_init, meta_4_critic_update = Meta_update('pop_4_critic', 'meta_4_critic', popsize, epsilon)
meta_5_actor_init, meta_5_actor_update = Meta_update('pop_5_actor', 'meta_5_actor', popsize, epsilon)
meta_5_critic_init, meta_5_critic_update = Meta_update('pop_5_critic', 'meta_5_critic', popsize, epsilon)
meta_6_actor_init, meta_6_actor_update = Meta_update('pop_6_actor', 'meta_6_actor', popsize, epsilon)
meta_6_critic_init, meta_6_critic_update = Meta_update('pop_6_critic', 'meta_6_critic', popsize, epsilon)
meta_7_actor_init, meta_7_actor_update = Meta_update('pop_7_actor', 'meta_7_actor', popsize, epsilon)
meta_7_critic_init, meta_7_critic_update = Meta_update('pop_7_critic', 'meta_7_critic', popsize, epsilon)
meta_8_actor_init, meta_8_actor_update = Meta_update('pop_8_actor', 'meta_8_actor', popsize, epsilon)
meta_8_critic_init, meta_8_critic_update = Meta_update('pop_8_critic', 'meta_8_critic', popsize, epsilon)
meta_9_actor_init, meta_9_actor_update = Meta_update('pop_9_actor', 'meta_9_actor', popsize, epsilon)
meta_9_critic_init, meta_9_critic_update = Meta_update('pop_9_critic', 'meta_9_critic', popsize, epsilon)
meta_10_actor_init, meta_10_actor_update = Meta_update('pop_10_actor', 'meta_10_actor', popsize, epsilon)
meta_10_critic_init, meta_10_critic_update = Meta_update('pop_10_critic', 'meta_10_critic', popsize, epsilon)

f1 = FL_Agent()
f2 = FL_Agent()
f3 = FL_Agent()
f4 = FL_Agent()
f5 = FL_Agent()
f6 = FL_Agent()
f7 = FL_Agent()
f8 = FL_Agent()
f9 = FL_Agent()
f10 = FL_Agent()

M1 = Memory(MEMORY_CAPACITY)
M2 = Memory(MEMORY_CAPACITY)
M3 = Memory(MEMORY_CAPACITY)
M4 = Memory(MEMORY_CAPACITY)
M5 = Memory(MEMORY_CAPACITY)
M6 = Memory(MEMORY_CAPACITY)
M7 = Memory(MEMORY_CAPACITY)
M8 = Memory(MEMORY_CAPACITY)
M9 = Memory(MEMORY_CAPACITY)
M10 = Memory(MEMORY_CAPACITY)


def UserChoose(char_data1, char_data2, users_set, choose):
    global char_data11
    global char_data22
    var = 1
    count = 0
    start_flag = False
    reward_all = []
    eps_add =[]
    accuracy_add = []
    loss_local_add = []
    loss_global_add = []
    PriceRecord = []
    T_meta = 10
    popsize = 10

    PriceRecord = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    PriceRecord = PriceRecord.repeat(10)
    PriceRecord = PriceRecord.reshape(10, 10)
    start_flag = False
    decison_making = []
    Staleness = []
    sess.run(tf.global_variables_initializer())
    sess.run([agent_1_actor_target_init, agent_1_critic_target_init, agent_2_actor_target_init,
                      agent_2_critic_target_init, agent_3_actor_target_init, agent_3_critic_target_init,
                      agent_4_actor_target_init,
                      agent_4_critic_target_init, agent_5_actor_target_init, agent_5_critic_target_init,
                      agent_6_actor_target_init,
                      agent_6_critic_target_init, agent_7_actor_target_init, agent_7_critic_target_init,
                      agent_8_actor_target_init,
                      agent_8_critic_target_init, agent_9_actor_target_init, agent_9_critic_target_init,
                      agent_10_actor_target_init,
                      agent_10_critic_target_init])

    for p in range(popsize):
        p=9
        Price = PriceRecord[p]
        start_flag = False
        # M1 = Memory(MEMORY_CAPACITY)
        # M2 = Memory(MEMORY_CAPACITY)
        # M3 = Memory(MEMORY_CAPACITY)
        # M4 = Memory(MEMORY_CAPACITY)
        # M5 = Memory(MEMORY_CAPACITY)
        # M6 = Memory(MEMORY_CAPACITY)
        # M7 = Memory(MEMORY_CAPACITY)
        # M8 = Memory(MEMORY_CAPACITY)
        # M9 = Memory(MEMORY_CAPACITY)
        # M10 = Memory(MEMORY_CAPACITY)
        # sess.run(tf.global_variables_initializer())
        # sess.run([agent_1_actor_target_init, agent_1_critic_target_init, agent_2_actor_target_init,
        #                   agent_2_critic_target_init, agent_3_actor_target_init, agent_3_critic_target_init,
        #                   agent_4_actor_target_init,
        #                   agent_4_critic_target_init, agent_5_actor_target_init, agent_5_critic_target_init,
        #                   agent_6_actor_target_init,
        #                   agent_6_critic_target_init, agent_7_actor_target_init, agent_7_critic_target_init,
        #                   agent_8_actor_target_init,
        #                   agent_8_critic_target_init, agent_9_actor_target_init, agent_9_critic_target_init,
        #                   agent_10_actor_target_init,
        #                   agent_10_critic_target_init])
        for eps in range(MAX_EPISODES):
            args, args.device, dict_users, dataset_train, dataset_test, net_glob, w_glob, loss_train, Power_train, Optim_train, cv_loss, cv_acc, val_loss_pre, counter, net_best, best_loss, val_acc_list, net_list = initial()
            char_data1 = np.ones(10)  # For staleness
            char_data2 = np.ones(10) * 2
            char_data11 = char_data1
            char_data22 = char_data2
            Datasize = [25, 9, 32, 6, 45, 20, 1, 12, 6, 11]

            UserPower = [0.4797846374733109, 0.10318370487752862, 0.24691822115565296, 0.1867959886387748,
                     0.34249845996578077, 0.168317423981345, 0.06423717394565366, 0.21075237584492076,
                     0.2642483615848909, 0.31922299207716565]
            psi = 0.5
            tau = 0.8
            f1.init(Datasize[0], UserPower[0])
            f2.init(Datasize[1], UserPower[1])
            f3.init(Datasize[2], UserPower[2])
            f4.init(Datasize[3], UserPower[3])
            f5.init(Datasize[4], UserPower[4])
            f6.init(Datasize[5], UserPower[5])
            f7.init(Datasize[6], UserPower[6])
            f8.init(Datasize[7], UserPower[7])
            f9.init(Datasize[8], UserPower[8])
            f10.init(Datasize[9], UserPower[9])
            ene_eps = 0
            rew1_eps, rew2_eps, rew3_eps = 0, 0, 0
            rew4_eps, rew5_eps, rew6_eps = 0, 0, 0
            rew7_eps, rew8_eps, rew9_eps, rew10_eps = 0, 0, 0, 0
            for step in range(1, MAX_EP_STEPS + 1, 1):
                if step==1:
                    char_data3 = np.ones(10) * 3
                    char_data33 = char_data3
                    qulity = char_data33 - char_data22

                obs_f1 = np.concatenate(([qulity[0], char_data22[0], Datasize[0], UserPower[0], Price[0]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f2 = np.concatenate(([qulity[1], char_data22[1], Datasize[1], UserPower[1], Price[1]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f3 = np.concatenate(([qulity[2], char_data22[2], Datasize[2], UserPower[2], Price[2]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f4 = np.concatenate(([qulity[3], char_data22[3], Datasize[3], UserPower[3], Price[3]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f5 = np.concatenate(([qulity[4], char_data22[4], Datasize[4], UserPower[4], Price[4]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f6 = np.concatenate(([qulity[5], char_data22[5], Datasize[5], UserPower[5], Price[5]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f7 = np.concatenate(([qulity[6], char_data22[6], Datasize[6], UserPower[6], Price[6]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f8 = np.concatenate(([qulity[7], char_data22[7], Datasize[7], UserPower[7], Price[7]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f9 = np.concatenate(([qulity[8], char_data22[8], Datasize[8], UserPower[8], Price[8]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f10 = np.concatenate(([qulity[9], char_data22[9], Datasize[9], UserPower[9], Price[9]], qulity,
                                      char_data22, Datasize, UserPower, Price))
                state = np.concatenate((obs_f1, obs_f2, obs_f3, obs_f4, obs_f5, obs_f6, obs_f7, obs_f8, obs_f9, obs_f10))

                action_f1_test = np.clip(agent_1.action(obs=[obs_f1], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f1_test > 0:
                    action_f1 = 1
                else:
                    action_f1 = 0

                action_f2_test = np.clip(agent_2.action(obs=[obs_f2], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f2_test > 0:
                    action_f2 = 1
                else:
                    action_f2 = 0

                action_f3_test = np.clip(agent_3.action(obs=[obs_f3], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f3_test > 0:
                    action_f3 = 1
                else:
                    action_f3 = 0

                action_f4_test = np.clip(agent_4.action(obs=[obs_f4], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f4_test > 0:
                    action_f4 = 1
                else:
                    action_f4 = 0

                action_f5_test = np.clip(agent_5.action(obs=[obs_f5], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f5_test > 0:
                    action_f5 = 1
                else:
                    action_f5 = 0

                action_f6_test = np.clip(agent_6.action(obs=[obs_f6], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f6_test > 0:
                    action_f6 = 1
                else:
                    action_f6 = 0

                action_f7_test = np.clip(agent_7.action(obs=[obs_f7], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f7_test > 0:
                    action_f7 = 1
                else:
                    action_f7 = 0

                action_f8_test = np.clip(agent_8.action(obs=[obs_f8], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f8_test > 0:
                    action_f8 = 1
                else:
                    action_f8 = 0

                action_f9_test = np.clip(agent_9.action(obs=[obs_f9], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f9_test > 0:
                    action_f9 = 1
                else:
                    action_f9 = 0

                action_f10_test = np.clip(agent_10.action(obs=[obs_f10], sess=sess) + np.random.randn(1) * var, -1, 1)
                if action_f10_test > 0:
                    action_f10 = 1
                else:
                    action_f10 = 0

                if action_f1 + action_f2 + action_f3 + action_f4 + action_f5 + action_f6 + action_f7 + action_f8 + action_f9 + action_f10 <= 1:
                    char_data1 = char_data1 + 1
                    action_f1, action_f10 = 1
                elif action_f1 + action_f2 + action_f3 + action_f4 + action_f5 + action_f6 + action_f7 + action_f8 + action_f9 + action_f10 >= 9:
                    # action_c = [action_f1, action_f2, action_f3, action_f4, action_f5, action_f6, action_f7,
                    #                 action_f8,action_f9, action_f10]
                    action_c = np.ones(10)
                    jj = random.sample(range(0, 10), 8)
                    for m in range(len(jj)):
                        action_c[jj[m]] = 0
                    action_f1, action_f2, action_f3, action_f4, action_f5, action_f6, action_f7, action_f8, action_f9, action_f10 = \
                            action_c[0], action_c[1], action_c[2], action_c[3], action_c[4], action_c[5], action_c[6], \
                            action_c[7], action_c[8], action_c[9]
                action_all = [action_f1, action_f2, action_f3, action_f4, action_f5, action_f6, action_f7, action_f8,
                          action_f9, action_f10]

                rew1 = (Price[0]*(1-1/math.exp((psi*(Datasize[0]*qulity[0])))) - UserPower[0]) * action_f1 * 10
                rew2 = (Price[1]*(1-1/math.exp((psi*(Datasize[1]*qulity[1])))) - UserPower[1]) * action_f2 * 10
                rew3 = (Price[2]*(1-1/math.exp((psi*(Datasize[2]*qulity[2])))) - UserPower[2]) * action_f3 * 10
                rew4 = (Price[3]*(1-1/math.exp((psi*(Datasize[3]*qulity[3])))) - UserPower[3]) * action_f4 * 10
                rew5 = (Price[4]*(1-1/math.exp((psi*(Datasize[4]*qulity[4])))) - UserPower[4]) * action_f5 * 10
                rew6 = (Price[5]*(1-1/math.exp((psi*(Datasize[5]*qulity[5])))) - UserPower[5]) * action_f6 * 10
                rew7 = (Price[6]*(1-1/math.exp((psi*(Datasize[6]*qulity[6])))) - UserPower[6]) * action_f7 * 10
                rew8 = (Price[7]*(1-1/math.exp((psi*(Datasize[7]*qulity[7])))) - UserPower[7]) * action_f8 * 10
                rew9 = (Price[8]*(1-1/math.exp((psi*(Datasize[8]*qulity[8])))) - UserPower[8]) * action_f9 * 10
                rew10 = (Price[9]*(1-1/math.exp((psi*(Datasize[9]*qulity[9])))) - UserPower[9]) * action_f10 * 10
                char_data3 = copy.deepcopy(char_data22)
                char_data33 = char_data3
                w_locals, loss_locals = [], []
                idxs_users = []
                for i in list(users_set):
                    if action_all[i] == 1:
                        idxs_users.append(i)

                for idx in list(set(users_set) - set(idxs_users)):  # staleness
                    char_data1[idx] = char_data1[idx] + 1

                for idx in idxs_users:
                    w_locals, loss_locals, char_data1, char_data2 = local_one(w_locals=w_locals,
                                                                              loss_locals=loss_locals,
                                                                              args=args,
                                                                              dataset_train=dataset_train,
                                                                              net_glob=net_glob,
                                                                              dict_users=dict_users,
                                                                              idx=idx, char_data1=char_data1,
                                                                              char_data2=char_data2)

                # update global weights
                w_glob = FedAvg(w_locals, idxs_users)
                loss_avg, acc_global, loss_global = caculate(net_glob=net_glob, args=args,
                                                             dataset_train=dataset_train,
                                                             w_locals=w_locals, idxs_users=idxs_users,
                                                             loss_locals=loss_locals)
                accuracy = acc_global.item()
                rew1_eps += rew1
                rew2_eps += rew2
                rew3_eps += rew3
                rew4_eps += rew4
                rew5_eps += rew5
                rew6_eps += rew6
                rew7_eps += rew7
                rew8_eps += rew8
                rew9_eps += rew9
                rew10_eps += rew10
                action_all = [action_f1, action_f2, action_f3, action_f4, action_f5, action_f6, action_f7, action_f8,
                          action_f9, action_f10]
                char_data11 = char_data1
                char_data22 = char_data2

                qulity = char_data33 - char_data22

                obs_f1_ = np.concatenate(([qulity[0], char_data22[0], Datasize[0], UserPower[0], Price[0]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f2_ = np.concatenate(([qulity[1], char_data22[1], Datasize[1], UserPower[1], Price[1]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f3_ = np.concatenate(([qulity[2], char_data22[2], Datasize[2], UserPower[2], Price[2]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f4_ = np.concatenate(([qulity[3], char_data22[3], Datasize[3], UserPower[3], Price[3]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f5_ = np.concatenate(([qulity[4], char_data22[4], Datasize[4], UserPower[4], Price[4]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f6_ = np.concatenate(([qulity[5], char_data22[5], Datasize[5], UserPower[5], Price[5]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f7_ = np.concatenate(([qulity[6], char_data22[6], Datasize[6], UserPower[6], Price[6]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f8_ = np.concatenate(([qulity[7], char_data22[7], Datasize[7], UserPower[7], Price[7]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f9_ = np.concatenate(([qulity[8], char_data22[8], Datasize[8], UserPower[8], Price[8]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                obs_f10_ = np.concatenate(([qulity[9], char_data22[9], Datasize[9], UserPower[9], Price[9]], qulity,
                                     char_data22, Datasize, UserPower, Price))
                state_ = np.concatenate(
                (obs_f1_, obs_f2_, obs_f3_, obs_f4_, obs_f5_, obs_f6_, obs_f7_, obs_f8_, obs_f9_, obs_f10_))
                transition_u = np.concatenate((state, [action_f1, action_f2, action_f3, action_f4, action_f5, action_f6,
                                                   action_f7, action_f8, action_f9, action_f10], [rew1], [rew2], [rew3],
                                           [rew4], [rew5], [rew6], [rew7], [rew8], [rew9], [rew10], state_), axis=0)
                M1.store(np.max(M1.tree.tree[-M1.tree.capacity:]), transition_u)
                M2.store(np.max(M2.tree.tree[-M2.tree.capacity:]), transition_u)
                M3.store(np.max(M3.tree.tree[-M3.tree.capacity:]), transition_u)
                M4.store(np.max(M4.tree.tree[-M4.tree.capacity:]), transition_u)
                M5.store(np.max(M5.tree.tree[-M5.tree.capacity:]), transition_u)
                M6.store(np.max(M6.tree.tree[-M6.tree.capacity:]), transition_u)
                M7.store(np.max(M7.tree.tree[-M7.tree.capacity:]), transition_u)
                M8.store(np.max(M8.tree.tree[-M8.tree.capacity:]), transition_u)
                M9.store(np.max(M9.tree.tree[-M9.tree.capacity:]), transition_u)
                M10.store(np.max(M10.tree.tree[-M10.tree.capacity:]), transition_u)
                count += 1

                if (eps + 1) >= 7:
                    if var > 0.05:
                        var *= .9995
                    start_flag = True

                    # Agent_1
                    tree_idx, b_M, ISWeights = M1.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, \
                    f4_b_action, f5_b_action, f6_b_action, f7_b_action, \
                    f8_b_action, f9_b_action, f10_b_action, f1_b_reward, \
                    f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, \
                    f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, \
                    f10_b_reward, b_state_next, f1_b_action_next, \
                    f2_b_action_next, f3_b_action_next, f4_b_action_next, \
                    f5_b_action_next, f6_b_action_next, f7_b_action_next, \
                    f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(b_M)
                    other_action = np.concatenate((f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action,
                                               f7_b_action, f8_b_action, f9_b_action, f10_b_action), axis=1)
                    other_action_next = np.concatenate((f2_b_action_next, f3_b_action_next, f4_b_action_next,
                                                    f5_b_action_next, f6_b_action_next, f7_b_action_next,
                                                    f8_b_action_next, f9_b_action_next, f10_b_action_next), axis=1)
                    target_fl = f1_b_reward.reshape(-1, 1) + gamma * agent_1_target.Q(state=b_state_next,
                                                                                  action=f1_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_1.train_actor(state=f_state, obs=f_state[:, 0:55], other_action=other_action, sess=sess)
                    abs_td = agent_1.train_critic(state=f_state, action=f1_b_action, other_action=other_action,
                                              target=target_fl, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M1.update(idx, abs_td[i])
                    # Agent_2
                    tree_idx, b_M, ISWeights = M2.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                    b_M)
                    other_action = np.concatenate((f1_b_action, f3_b_action, f4_b_action, f5_b_action,
                                               f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f3_b_action_next,
                                                    f4_b_action_next, f5_b_action_next, f6_b_action_next,
                                                    f7_b_action_next, f8_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f2 = f2_b_reward.reshape(-1, 1) + gamma * agent_2_target.Q(state=b_state_next,
                                                                                  action=f2_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)#转换成一列
                    agent_2.train_actor(state=f_state, obs=f_state[:, 55:110], other_action=other_action, sess=sess)
                    abs_td = agent_2.train_critic(state=f_state, action=f2_b_action, other_action=other_action,
                                              target=target_f2, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M2.update(idx, abs_td[i])

                    # Agent_3
                    tree_idx, b_M, ISWeights = M3.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                    b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f4_b_action, f5_b_action,
                                               f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f4_b_action_next, f5_b_action_next, f6_b_action_next,
                                                    f7_b_action_next, f8_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f3 = f3_b_reward.reshape(-1, 1) + gamma * agent_3_target.Q(state=b_state_next,
                                                                                  action=f3_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_3.train_actor(state=f_state, obs=f_state[:, 110:165], other_action=other_action, sess=sess)
                    abs_td = agent_3.train_critic(state=f_state, action=f3_b_action, other_action=other_action,
                                              target=target_f3, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M3.update(idx, abs_td[i])

                    # Agent_4
                    tree_idx, b_M, ISWeights = M4.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f5_b_action,
                                               f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f5_b_action_next, f6_b_action_next,
                                                    f7_b_action_next, f8_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f4 = f4_b_reward.reshape(-1, 1) + gamma * agent_4_target.Q(state=b_state_next,
                                                                                  action=f4_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_4.train_actor(state=f_state, obs=f_state[:, 165:220], other_action=other_action, sess=sess)
                    abs_td = agent_4.train_critic(state=f_state, action=f4_b_action, other_action=other_action,
                                              target=target_f4, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M4.update(idx, abs_td[i])

                    # Agent_5
                    tree_idx, b_M, ISWeights = M5.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, u1_b_reward, u2_b_reward, u3_b_reward, u4_b_reward, u5_b_reward, u6_b_reward, u7_b_reward, u8_b_reward, u9_b_reward, u10_b_reward, b_state_next, u1_b_action_next, u2_b_action_next, u3_b_action_next, u4_b_action_next, u5_b_action_next, u6_b_action_next, u7_b_action_next, u8_b_action_next, u9_b_action_next, u10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f4_b_action,
                                               f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f4_b_action_next, f6_b_action_next,
                                                    f7_b_action_next, f8_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f5 = f5_b_reward.reshape(-1, 1) + gamma * agent_5_target.Q(state=b_state_next,
                                                                                  action=f5_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_5.train_actor(state=f_state, obs=f_state[:, 220:275], other_action=other_action, sess=sess)
                    abs_td = agent_5.train_critic(state=f_state, action=f5_b_action, other_action=other_action,
                                              target=target_f5, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M5.update(idx, abs_td[i])

                    # Agent_6
                    tree_idx, b_M, ISWeights = M6.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f4_b_action,
                                               f5_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f4_b_action_next, f6_b_action_next,
                                                    f7_b_action_next, f8_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f6 = f6_b_reward.reshape(-1, 1) + gamma * agent_6_target.Q(state=b_state_next,
                                                                                  action=f6_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_6.train_actor(state=f_state, obs=f_state[:, 275:330], other_action=other_action, sess=sess)
                    abs_td = agent_6.train_critic(state=f_state, action=f6_b_action, other_action=other_action,
                                              target=target_f6, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M6.update(idx, abs_td[i])

                    # Agent_7
                    tree_idx, b_M, ISWeights = M7.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f4_b_action,
                                               f5_b_action, f6_b_action, f8_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f4_b_action_next, f5_b_action_next,
                                                    f6_b_action_next, f8_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f7 = f7_b_reward.reshape(-1, 1) + gamma * agent_7_target.Q(state=b_state_next,
                                                                                  action=f7_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_7.train_actor(state=f_state, obs=f_state[:, 330:385], other_action=other_action, sess=sess)
                    abs_td = agent_7.train_critic(state=f_state, action=f7_b_action, other_action=other_action,
                                              target=target_f7, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M7.update(idx, abs_td[i])

                    # Agent_8
                    tree_idx, b_M, ISWeights = M8.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f4_b_action,
                                               f5_b_action, f6_b_action, f7_b_action, f9_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f4_b_action_next, f5_b_action_next,
                                                    f6_b_action_next, f7_b_action_next, f9_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f8 = f8_b_reward.reshape(-1, 1) + gamma * agent_8_target.Q(state=b_state_next,
                                                                                  action=f8_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_8.train_actor(state=f_state, obs=f_state[:, 385:440], other_action=other_action, sess=sess)
                    abs_td = agent_8.train_critic(state=f_state, action=f8_b_action, other_action=other_action,
                                              target=target_f8, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M8.update(idx, abs_td[i])

                    # Agent_9
                    tree_idx, b_M, ISWeights = M9.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f4_b_action,
                                               f5_b_action, f6_b_action, f7_b_action, f8_b_action, f10_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f4_b_action_next, f5_b_action_next,
                                                    f6_b_action_next, f7_b_action_next, f8_b_action_next,
                                                    f10_b_action_next), axis=1)
                    target_f9 = f9_b_reward.reshape(-1, 1) + gamma * agent_9_target.Q(state=b_state_next,
                                                                                  action=f9_b_action_next,
                                                                                  other_action=other_action_next,
                                                                                  sess=sess)
                    agent_9.train_actor(state=f_state, obs=f_state[:, 440:495], other_action=other_action, sess=sess)
                    abs_td = agent_9.train_critic(state=f_state, action=f9_b_action, other_action=other_action,
                                              target=target_f9, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M9.update(idx, abs_td[i])

                    # Agent_10
                    tree_idx, b_M, ISWeights = M10.prio_sample(BATCH_SIZE)
                    f_state, f1_b_action, f2_b_action, f3_b_action, f4_b_action, f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action, f10_b_action, f1_b_reward, f2_b_reward, f3_b_reward, f4_b_reward, f5_b_reward, f6_b_reward, f7_b_reward, f8_b_reward, f9_b_reward, f10_b_reward, b_state_next, f1_b_action_next, f2_b_action_next, f3_b_action_next, f4_b_action_next, f5_b_action_next, f6_b_action_next, f7_b_action_next, f8_b_action_next, f9_b_action_next, f10_b_action_next = necessary(
                        b_M)
                    other_action = np.concatenate((f1_b_action, f2_b_action, f3_b_action, f4_b_action,
                                               f5_b_action, f6_b_action, f7_b_action, f8_b_action, f9_b_action),
                                              axis=1)
                    other_action_next = np.concatenate((f1_b_action_next, f2_b_action_next,
                                                    f3_b_action_next, f4_b_action_next, f5_b_action_next,
                                                    f6_b_action_next, f7_b_action_next, f8_b_action_next,
                                                    f9_b_action_next), axis=1)
                    target_fl0 = f10_b_reward.reshape(-1, 1) + gamma * agent_10_target.Q(state=b_state_next,
                                                                                    action=f10_b_action_next,
                                                                                    other_action=other_action_next,
                                                                                    sess=sess)
                    agent_10.train_actor(state=f_state, obs=f_state[:, 495:550], other_action=other_action, sess=sess)
                    abs_td = agent_10.train_critic(state=f_state, action=f10_b_action, other_action=other_action,
                                               target=target_fl0, sess=sess, ISWeights=ISWeights)
                    for i in range(len(tree_idx)):
                        idx = tree_idx[i]
                        M10.update(idx, abs_td[i])

                    sess.run([agent_1_actor_target_update, agent_1_critic_target_update])
                    sess.run([agent_2_actor_target_update, agent_2_critic_target_update])
                    sess.run([agent_3_actor_target_update, agent_3_critic_target_update])
                    sess.run([agent_4_actor_target_update, agent_4_critic_target_update])
                    sess.run([agent_5_actor_target_update, agent_5_critic_target_update])
                    sess.run([agent_6_actor_target_update, agent_6_critic_target_update])
                    sess.run([agent_7_actor_target_update, agent_7_critic_target_update])
                    sess.run([agent_8_actor_target_update, agent_8_critic_target_update])
                    sess.run([agent_9_actor_target_update, agent_9_critic_target_update])
                    sess.run([agent_10_actor_target_update, agent_10_critic_target_update])
            print(' p:%s, eps:%s, step:%s, r_all:%s, loss_avg:%s, acc_global:%s, start_flag:%s' % (p, eps, step, (
                        rew1_eps + rew2_eps + rew3_eps + rew4_eps + rew5_eps + rew6_eps + rew7_eps + rew8_eps + rew9_eps + rew10_eps) / 10,
                                                                                        loss_avg, accuracy,
                                                                                             start_flag))
            print(action_all)
            print('Price:%s' % Price)



            reward_all.append((rew1_eps + rew2_eps + rew3_eps + rew4_eps + rew5_eps + rew6_eps + rew7_eps + rew8_eps + rew9_eps + rew10_eps) / 10)
            reward_eps = (rew1_eps + rew2_eps + rew3_eps + rew4_eps + rew5_eps + rew6_eps + rew7_eps + rew8_eps + rew9_eps + rew10_eps) / 10

            eps_add.append(eps)
            mm=[reward_all, eps_add]
            np.savetxt('C:/Users/USER/Desktop/Result/3.18/reward_2.txt', mm, delimiter=',')
            accuracy_add.append(accuracy)
            loss_local_add.append(loss_avg)
            loss_global_add.append(loss_global)

    sess.run([pop_1_actor_update, pop_1_critic_update, pop_2_actor_update, pop_2_critic_update,
                  pop_3_actor_update, pop_3_critic_update, pop_4_actor_update, pop_4_critic_update,
                  pop_5_actor_update, pop_5_critic_update, pop_6_actor_update, pop_6_critic_update,
                  pop_7_actor_update, pop_7_critic_update, pop_8_actor_update, pop_8_critic_update,
                  pop_9_actor_update, pop_9_critic_update, pop_10_actor_update, pop_10_critic_update])
    decison_making.append(action_all)
    Staleness.append(char_data11)


    sess.run([meta_1_actor_update, meta_1_critic_update, meta_2_actor_update, meta_2_critic_update,
                  meta_3_actor_update, meta_3_critic_update, meta_4_actor_update, meta_4_critic_update,
                  meta_5_actor_update, meta_5_critic_update, meta_6_actor_update, meta_6_critic_update,
                  meta_7_actor_update, meta_7_critic_update, meta_8_actor_update, meta_8_critic_update,
                  meta_9_actor_update, meta_9_critic_update, meta_10_actor_update, meta_10_critic_update,])
    if p==9:
        meta_1.save(name='_meta_1_' + str(p + 1), sess=sess)
        meta_2.save(name='_meta_2_' + str(p + 1), sess=sess)
        meta_3.save(name='_meta_3_' + str(p + 1), sess=sess)
        meta_4.save(name='_meta_4_' + str(p + 1), sess=sess)
        meta_5.save(name='_meta_5_' + str(p + 1), sess=sess)
        meta_6.save(name='_meta_6_' + str(p + 1), sess=sess)
        meta_7.save(name='_meta_7_' + str(p + 1), sess=sess)
        meta_8.save(name='_meta_8_' + str(p + 1), sess=sess)
        meta_9.save(name='_meta_9_' + str(p + 1), sess=sess)
        meta_10.save(name='_meta_10_' + str(p + 1), sess=sess)


#
if __name__ == '__main__':
    char_data1 = np.ones(10)  # For staleness
    char_data2 = np.ones(10) * 10
    UserSetting = [i for i in range(10)]
    choice_all = 4  # For loss
    RecordF = []
    UserChoose(char_data1, char_data2, UserSetting, choice_all)#保存模型
