

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from agent_dir.agent import Agent
import scipy
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import Activation


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


np.random.seed(46)


transfer_action_dict = {}
transfer_action_dict[0] = 0
transfer_action_dict[1] = 2
transfer_action_dict[2] = 3


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)




class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        
        self.env = env
        self.args = args
        
        self.state_size = 80 * 80
        self.action_num = env.action_space.n
        
        self.gamma = 0.99
        self.lr = 1e-4
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        
        self.state = None
        self.current_state = None
        self.previous_state = None
        
        self.restore_training = True

        self.model_path = './pg_model.h5'

        self.episode = 1
        self.episode_path = './pg_episode'

        self.epi_scores = []
        self.epi_scores_path = './pg_epi_scores'

        self.avg_epi_scores = []
        self.avg_epi_scores_path = './pg_avg_epi_scores'
        self.avg_epi_scores_img_path = './pg_avg_epi_scores.png'
        
        self.model = self.pg_model()
        self.model.summary()
        
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.load_weights(self.model_path)

        self.test_previous_action = 0
    
    
    def pg_model(self):
        
        
        model = Sequential()
        model.add(Reshape((80, 80, 1), input_shape = (self.state_size,)))
        
        model.add(Conv2D(filters = 16, kernel_size = (8, 8), strides = (4, 4), 
                                        padding ='same', kernel_initializer = 'he_uniform'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters = 32, kernel_size = (4, 4), strides = (2, 2), 
                                        padding ='same', kernel_initializer = 'he_uniform'))
        model.add(Activation('relu'))
        
        model.add(Flatten())
        
        
        model.add(Dense(128, kernel_initializer = 'he_uniform'))
        model.add(Activation('relu'))
        
        
         
        # model.add(Dense(self.action_num))
        model.add(Dense(3))

        model.add(Activation('softmax'))
        
        opt = Adam(lr = self.lr)
        # opt = RMSprop(lr = self.lr, decay = 0.99)

        model.compile(
                loss = 'categorical_crossentropy',
                optimizer = opt
                )
        
        return model


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        if self.restore_training:

            self.model.load_weights(self.model_path)

            with open(self.episode_path, 'rb') as fp:
                self.episode = pickle.load(fp)
            self.episode += 1

            with open(self.epi_scores_path, 'rb') as fp:
                self.epi_scores = pickle.load(fp)

            with open(self.avg_epi_scores_path, 'rb') as fp:
                self.avg_epi_scores = pickle.load(fp)
        
        
        state = self.env.reset()
        self.previous_state = None
        score = 0
        
        
        my_score = 0
        rival_score = 0
        
        avg_epi_score = 0
        

        while True:
            
            action, prob = self.make_action(state, test = False)
            state, reward, done, info = self.env.step(action)
            score += reward

            # information record
            y = np.zeros([3])
            if action == 0 or action == 1:
                y[0] = 1
            elif action == 2 or action == 4:
                y[1] = 1
            else:
                y[2] = 1

            y = np.asarray(y)
            y = y.astype('float32')

            gradient = y - prob

            
            self.states.append(self.state)
            self.rewards.append(reward)
            self.gradients.append(gradient)
            
            if reward == 1:
                my_score += 1
            elif reward == -1:
                rival_score += 1
            
            
            score_table = [my_score, rival_score]
    
            if done:

                self.epi_scores.append(score)
                
                if self.episode >= 30:
                    avg_epi_score = np.mean(self.epi_scores[-30:])
                    self.avg_epi_scores.append(avg_epi_score)

                    
                print('\n')
                print('Episode', self.episode, ':', score_table)
                print('Episode:', self.episode, '-', 'Score:', score)
                print('Episode:', self.episode, '-','AVG_Score:', avg_epi_score)
                print('\n')
                
                self.train_on_batch()
                
                score = 0
                my_score = 0
                rival_score = 0
                
                state = self.env.reset()
                self.previous_state = None
                
                if self.episode % 10 == 0:

                    print('\n')
                    print('Saving Model')
                    print('\n')

                    self.model.save_weights(self.model_path)

                    with open(self.episode_path, 'wb') as fp:
                        pickle.dump(self.episode, fp)

                    with open(self.epi_scores_path, 'wb') as fp:
                        pickle.dump(self.epi_scores, fp)

                    with open(self.avg_epi_scores_path, 'wb') as fp:
                        pickle.dump(self.avg_epi_scores, fp)


                    # if self.episode >= 30:

                    #     epi_x = np.asarray(range(len(self.avg_epi_scores))) + 30

                    #     plt.plot(epi_x, self.avg_epi_scores)
                    #     plt.title('avg_epi_scores')
                    #     plt.xlabel('episode')
                    #     plt.ylabel('score')
                    #     plt.ylim(-25, 25)
                    #     plt.savefig(self.avg_epi_scores_img_path)
                    #     plt.clf()

                self.episode += 1
        
        
        # pass
    
    
    def train_on_batch(self,):

        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)

        # discount
        the_rewards = np.zeros_like(rewards)
        credit = 0
        for i in reversed(range(len(rewards))):
            if rewards[i] != 0:
                credit = 0
            credit = self.gamma * credit + rewards[i]
            the_rewards[i] = credit

        rewards = the_rewards


        # normailize
        mean_rewards = np.mean(rewards)
        std_rewards = np.std(rewards)

        rewards -= mean_rewards
        rewards /= std_rewards

        gradients *= rewards

        X_batch = np.squeeze(np.vstack([self.states]))
        y_batch = self.probs + self.lr * np.squeeze(np.vstack([gradients]))

        self.model.train_on_batch(X_batch, y_batch)

        self.states = []
        self.probs = []
        self.gradients = []
        self.rewards = []


    def make_action(self, observation, test=True):
        
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        if test and np.random.random() <= 0.025:
            return self.test_previous_action
            # return self.env.get_random_action()

        
        
        self.current_state = prepro(observation)
        
        self.current_state = self.current_state.reshape(6400)

        if self.previous_state is not None:
            self.state = self.current_state - self.previous_state
        else:
            self.state = np.zeros(self.state_size)

        self.previous_state = self.current_state
        
        
        state = self.state.reshape([1, 6400])

        

        if test:

            action = np.argmax(self.model.predict(state, batch_size = 1))

            action = transfer_action_dict[action]

            self.test_previous_action = action

            return action

        

        prob = self.model.predict(state, batch_size = 1).flatten()
        self.probs.append(prob)

        the_prob = np.zeros(self.action_num)

        the_prob[0] = prob[0]
        the_prob[2] = prob[1]
        the_prob[3] = prob[2]

        the_prob = the_prob / np.sum(the_prob)

        action = np.random.choice(self.action_num, 1, p = the_prob)[0]
        
        return action, prob
        
        # return self.env.get_random_action()
