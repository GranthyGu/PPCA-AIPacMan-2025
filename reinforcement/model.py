
"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""

from torch.nn import Module
from torch.nn import  Linear
from torch import tensor, double, optim
from torch.nn.functional import relu, mse_loss



class DeepQNetwork(Module):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        super(DeepQNetwork, self).__init__()
        # Remember to set self.learning_rate, self.numTrainingGames,
        # and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.001
        self.numTrainingGames = 2000
        self.batch_size = 32
        self.fc1 = Linear(state_dim, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        "**END CODE"""
        self.double()


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        states_tensor = tensor(states)
        Q_target_tensor = tensor(Q_target)
        Q_pred = self.forward(states_tensor)
        loss = mse_loss(Q_pred, Q_target_tensor)
        return loss

    def forward(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(states))
        x = relu(self.fc2(x))
        Q_values = self.fc3(x)
        return Q_values
    
    def run(self, states):
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        You can look at the ML project for an idea of how to do this, but note that rather
        than iterating through a dataset, you should only be applying a single gradient step
        to the given datapoints.

        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        self.optimizer.zero_grad()
        loss = self.get_loss(states, Q_target)
        loss.backward()
        self.optimizer.step()