import rospy
import os
import json
import numpy as np
import random
import time
import sys
import matplotlib.pyplot as plt
import pickle as pl
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_4 import Env
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, TimeDistributed, GRU


EPISODES = 3001
ACTION_SKIP = 10

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/3001_20/stage_4_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(TimeDistributed(Dense(64, activation='relu', kernel_initializer='lecun_uniform'),input_shape=(4, 26)))

        model.add(TimeDistributed(Dense(64, activation='relu', kernel_initializer='lecun_uniform')))
        #model.add(Dropout(dropout))
        model.add(GRU(128))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))

        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))

        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, 4, 26))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)


        X_batch = np.empty((0, 4, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):

            states=mini_batch[i][0]
            actions=mini_batch[i][1]
            rewards=mini_batch[i][2]
            next_states=mini_batch[i][3]
            dones=mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, 4, 26))
            
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, 4, 26))

            else:
                next_target = self.model.predict(next_states.reshape(1, 4, 26))


            next_q_value = self.getQvalue(rewards, next_target, dones)

            #print(np.shape([states.copy()]))
            #print(np.shape(X_batch))
            #print(np.shape([states.copy()]))
            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            
            Y_sample = q_value.copy() 
            # Going to be three dimension
            #print(np.shape(Y_sample))


            Y_sample[0][actions] = next_q_value
            #print(np.shape(Y_batch))
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)
        
        #print(np.shape(Y_batch))
        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
plot_rwd=[]
plot_q_value=[]
if __name__ == '__main__':
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 26
    action_size = 5

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    #state_stack, action_stack, reward_stack, next_state_stack, done_stack
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()

        history = np.stack((state, state, state, state))

        history = np.reshape([history],(4, state_size))
        #print(np.shape(history))
        score = 0
        for t in range(agent.episode_step):
            #origin
            #action = agent.getAction(state)

            action = agent.getAction(history) # add

            next_state, reward, done = env.step(action)

            
            next_state = next_state.reshape(state_size) #add
            next_history = next_state.reshape(1, state_size) #add
            next_history = np.append(next_history, history[:3 ,:], axis = 0) #add
            #print(np.shape(history))
            #origin
            #agent.appendMemory(state, action, reward, next_state, done)

            agent.appendMemory(history, action, reward, next_history, done) #add

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            #origin
            #state = next_state

            history =  next_history #add
            
            get_action.data = [action, score, reward]

            if t%ACTION_SKIP ==0: # added 
                pub_get_action.publish(get_action) # publish action
            

            
            if e % 10 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))

                

                plot_rwd.append(score)
                plot_q_value.append(np.max(agent.q_value))
                plt.plot(np.linspace(1,e,e,dtype='int'),plot_rwd,color='blue',label='Score')
                plt.plot(np.linspace(1,e,e,dtype='int'),plot_q_value,color='red',label='Avg Q Value')
                plt.xlabel("EPISODE")
                plt.ylabel("VALUE")  
                if e == 1:
                    plt.legend()
                plt.grid(b=True,which='both',axis='both')
                plt.pause(0.01) 
                break
                
            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    plt.show()        
