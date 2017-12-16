from agent_dir.agent import Agent
import gym
import numpy as np
from keras.models import Model, Input
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras import losses
import pickle
# import scipy


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


np.random.seed(46)

class Transition(object):
    def __init__(self, observation, action, reward, done, previous_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.previous_observation = previous_observation


def preprocessing(observation):


    new_observation = np.moveaxis(observation, -1, 0)

    return new_observation



class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """



        super(Agent_DQN,self).__init__(env)

        

        ##################
        # YOUR CODE HERE #
        ##################

        self.env = env
        self.args = args

        self.action_num = 4
        self.batch_size = 32

        self.environments_steps = 5000
        self.experience_replay_size = 10000

        
        self.target_Q_update_freq = 10000
        self.online_Q_update_thresh = 10000
        self.gamma = 0.99
        self.action_repeat = 4
        self.lr = 1e-4
        
        self.start_exploration = 1.0
        self.end_exploration = 0.05
        self.exploration_frame = 1e6

        self.history_size = 4

        self.img_h = 84
        self.img_w = 84

        self.render = False

        self.online_Q = self.build_model()
        self.target_Q = self.copy_model(self.online_Q)

        self.online_Q.summary()

        self.transitions = []
        self.train_batches = []
        self.exploration_factor = self.start_exploration
        self.steps = 0

        self.restore_training = True

        self.online_Q_model_path = './online_Q_model.h5'
        self.target_Q_model_path = './target_Q_model.h5'

        self.episode = 0
        self.episode_path = './dqn_episode'

        self.epi_scores = []
        self.epi_scores_path = './dqn_epi_scores'

        self.avg_epi_scores = []
        self.avg_epi_scores_path = './dqn_avg_epi_scores'
        self.avg_epi_scores_img_path = './dqn_avg_epi_scores.png'

        

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

            self.online_Q.load_weights(self.online_Q_model_path)
            self.target_Q.load_weights(self.target_Q_model_path)

        elif self.restore_training:
            print('loading trained model')

            self.online_Q.load_weights(self.online_Q_model_path)
            self.target_Q.load_weights(self.target_Q_model_path)

            with open(self.episode_path, 'rb') as fp:
                self.episode = pickle.load(fp)

            with open(self.epi_scores_path, 'rb') as fp:
                self.epi_scores = pickle.load(fp)

            with open(self.avg_epi_scores_path, 'rb') as fp:
                self.avg_epi_scores = pickle.load(fp)

    def build_model(self):

        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding = 'same', input_shape=( self.history_size, self.img_h, self.img_w)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding = 'same'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding = 'same'))
        model.add(Flatten())
        model.add(Dense(512, activation=LeakyReLU()))
        model.add(Dense(self.action_num))

        opt = Adam(lr = self.lr)
        # opt = RMSprop(lr = self.lr, decay = 0.99)

        model.compile(
                loss = 'mse',
                optimizer = opt
                )

        return model


    def copy_model(self, model):
        model.save_weights('weights.h5')
        new_model = self.build_model()
        new_model.load_weights('weights.h5')
        return new_model

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

        while(1):


            observation = self.env.reset()

            
            observation = preprocessing(observation)
            previous_observation = observation[0]

            done = False
            total_reward = 0
            action = 0

            while not done:

                if self.render:
                    self.env.render()
                if self.steps % self.action_repeat == 0:
                    action = self.make_action(observation, test=False)

                observation, reward, done, info = self.env.step(action)

                total_reward += reward

                observation = preprocessing(observation)
                self.transitions.append(Transition(observation, action, reward, done, previous_observation))
                previous_observation = observation[0]

                if len(self.transitions) > self.experience_replay_size:
                    self.transitions = self.transitions[1:]

                self.steps += 1
                self.exploration_factor = np.max([self.end_exploration, self.exploration_factor - 1.0/self.exploration_frame])
                
                if len(self.transitions) >= self.environments_steps:

                    # macke batch
                    self.train_batches.append(np.random.choice(self.transitions, self.batch_size))

                    if len(self.train_batches) >= self.online_Q_update_thresh:
                        
                        # train batch
                        self.go_train_on_batch()

                        # clear batch
                        self.train_batches = []

                if self.steps % self.target_Q_update_freq == 0:
                    self.target_Q = self.copy_model(self.online_Q)
                    # print ("exploration_factor", self.exploration_factor)



                if done:
                    self.episode += 1
                    print(self.episode, total_reward)

                    self.epi_scores.append(total_reward)
                
                    if self.episode >= 30:
                        avg_epi_score = np.mean(self.epi_scores[-30:])
                        self.avg_epi_scores.append(avg_epi_score)

                    if self.episode % 250 == 0:

                        print('\n')
                        print('Saving Model')
                        print('\n')

                        self.online_Q.save_weights(self.online_Q_model_path)
                        self.target_Q.save_weights(self.target_Q_model_path)

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
                        #     plt.ylim(0, 1)
                        #     plt.savefig(self.avg_epi_scores_img_path)
                        #     plt.clf()

        
    def go_train_on_batch(self):

        print('train on batch ...')

        for batch in self.train_batches:


            sequences = []
            next_sequences = []

            for transition in batch:
                sequences.append([transition.previous_observation] + list(transition.observation[:-1]))
                next_sequences.append(list(transition.observation))

            targets = self.online_Q.predict(np.asarray(sequences))
            new_sequence_predictions = self.target_Q.predict(np.asarray(next_sequences))

            for index, prediction in enumerate(new_sequence_predictions):
                
                done = batch[index].done
                reward = batch[index].reward
                action = batch[index].action

                if not done:
                    max_q = np.max(prediction)
                else:
                    max_q = 0

                targets[index][action] = reward + (1 - done) * self.gamma * max_q

            self.online_Q.train_on_batch(np.asarray(next_sequences), targets)

        print('train on batch over')

        # pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        if test:

            observation = preprocessing(observation)

            x = np.asarray([observation])

            prediction = self.online_Q.predict(x)[0]

            return np.argmax(prediction)


        n = np.random.random()

        if n <= self.exploration_factor:
            return self.env.action_space.sample()

        x = np.asarray([observation])

        prediction = self.online_Q.predict(x)[0]

        return np.argmax(prediction)

        # return self.env.get_random_action()
