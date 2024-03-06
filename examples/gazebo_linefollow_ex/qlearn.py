import random
import pickle
import csv
import numpy as np


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        with open(filename+".pickle", 'rb') as f:
            loaded_q = pickle.load(f)

        self.q.update(loaded_q)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename+".pickle", 'wb') as f:
            pickle.dump(self.q, f, pickle.HIGHEST_PROTOCOL)

        print("Wrote to file: {}".format(filename+".pickle"))

        with open(filename+".csv", 'w') as f:
            writer = csv.writer(f)
            for key, value in self.q.items():
                writer.writerow([key, value])

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        ret_action = None
        if random.random() < self.epsilon:
            ret_action = random.choice(self.actions)
        else:
            ret_action = self.findBestAction(state)

        if return_q:
            return ret_action, self.getQ(state, ret_action)
        else:
            return ret_action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        self.q[(state1,action1)] = self.getQ(state1, action1) + self.alpha * (reward + self.gamma * self.findBestAction(state2, return_q=True) 
                                                  - self.getQ(state1, action1)) 
    
    def findBestAction(self, state, return_q=False):
        '''
        @brief returns the best action for a given state or the reward 
        value for that best action, if multiple actions have the same best reward,
        a random action of those with the best reward is returned
        '''
        bestAction = []
        bestQ = np.NINF
        for action in self.actions:
            q = self.getQ(state, action)
            if q > bestQ:
                bestQ = q
                bestAction.clear()
                bestAction.append(action)
            elif q == bestQ:
                bestAction.append(action)

        if return_q:
            return bestQ
        elif len(bestAction) > 1:
            return random.choice(bestAction)
        else:
            return bestAction[0]