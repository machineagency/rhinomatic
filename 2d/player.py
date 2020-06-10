import copy, random
from environment import Environment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, input_num_channels, grid_size):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(input_num_channels, 16, 4,
                               stride=2,
                               padding=2)
        self.conv2 = nn.Conv2d(16, 32, 4,
                               stride=2,
                               padding=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, 1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.GRID_SIZE = grid_size

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten_volume_to_vector(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def update(self, spec, next_states, actions, ys):
        """
        Takes a list of tetris states and the actions taken at that time, and
        YS which are state-action values calculated from the offline network.
        Predict the state-action values for the states in TETRIS_STATES,
        calculate loss against target YS, and update this (online) network.
        """
        # NOTE: using next_states to save time
        # next_states = []
        # temp_env = tetris.TetrisEnv()
        # for state, action in zip(states, actions):
        #     temp_env.set_state(state)
        #     next_state, reward, done, _ = temp_env.step(action)
        #     next_states.append(next_state)
        next_state_tensors = [self.calc_spec_state_tensor(spec, s)\
                                for s in next_states]
        next_state_batch = torch.stack(next_state_tensors, 0)
        q_preds_batch = self.predict_stack(next_state_batch)
        y_batch = torch.Tensor(ys).reshape((32, 1))

        self.optimizer.zero_grad()
        loss = self.criterion(q_preds_batch, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def OLD_update(self, x, a_indices, y):
        a_batch = a_indices.reshape((*(a_indices.shape), 1))
        y_batch = y.reshape((*(y.shape), 1))
        self.optimizer.zero_grad()
        prediction = self.predict(x, a_batch)
        loss = self.criterion(prediction, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_stack(self, x_stack):
        """
        x_stack.shape == (batch_size x 1 x 21 x 10)
        """
        return self.forward(x_stack)

    def predict_single(self, spec, state):
        env = Environment(self.GRID_SIZE, self.GRID_SIZE)
        env.set_spec(spec)
        actions = env.get_actions()
        print(actions)
        new_states = []
        for action in actions:
            new_state = env.peek_action(action)
            new_states.append(new_state)
        new_state_tensors = [self.calc_spec_state_tensor(spec, s) for s in new_states]
        new_state_stack = torch.stack(new_state_tensors, 0)
        q_preds = self.forward(new_state_stack)
        return (q_preds, actions)

    def predict_single_max_a(self, spec, state):
        q_preds, actions = self.predict_single(spec, state)
        return q_preds.max().item()

    def predict_single_argmax_a(self, spec, state):
        q_preds, actions = self.predict_single(spec, state)
        max_action_idx = q_preds.argmax()
        return actions[max_action_idx]

    def calc_spec_state_tensor(self, spec, state):
        stack = np.stack([spec, state], 2)
        t = torch.from_numpy(stack).reshape((2, self.GRID_SIZE,\
                self.GRID_SIZE)).type(torch.FloatTensor)
        return t

    def flatten_volume_to_vector(self, x):
        return x.view(-1, self.num_flat_features(x))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Buffer:
    def __init__(self, capacity, init_blank=False):
        self.lst = []
        self.capacity = capacity
        if init_blank:
            self.lst.append(self.gen_blank_exp())

    def gen_blank_exp(self):
        raise Exception()
        return (torch.zeros((1, 32, 32)).type(torch.FloatTensor), 0, 0,\
                torch.zeros((1, 32, 32)).type(torch.FloatTensor), False)

    def push(self, obj):
        """
        Push from left to right.
        """
        self.lst.insert(0, obj)
        if len(self.lst) > self.capacity:
            self.lst = self.lst[0:self.capacity]

    @property
    def most_recent(self):
        if len(self) == 0:
            return None
        return self.lst[0]

    @property
    def contents(self):
        return self.lst

    @property
    def full(self):
        return len(self.lst) == self.capacity

    def random_sample(self, sample_size):
        if sample_size > len(self.lst):
            return None
        return random.sample(self.lst, sample_size)

    def weighted_sample(self, sample_size, raw_weights):
        if sample_size > len(self.lst):
            return None
        n = len(raw_weights)
        weights = [w / n for w in raw_weights]
        return random.choices(self.lst, weights, k=sample_size)

    def __getitem__(self, idx):
        return self.lst[idx]

    def __len__(self):
        return len(self.lst)

class Player:
    def __init__(self):
        self.GRID_SIZE = 32
        self.NUM_CHANNELS = 2
        self.NUM_EPISODES = 1000
        self.REPLAY_CAPACITY = 1000
        self.MINIBATCH_SIZE = 32
        self.HISTORY_DEPTH = 1
        self.OFFLINE_UPDATE_FREQ = 100
        self.GAMMA = 0.99
        self.DEFAULT_ACTION_IDX = 4
        self.EPS_REDUCTION = 0.0001
        self.MIN_EPS = 0.1
        self.epsilon = 1.0
        self.frames_since_update = 0
        self.replay_buffer = Buffer(self.REPLAY_CAPACITY, init_blank=False)
        self.sample_weights = Buffer(self.REPLAY_CAPACITY, init_blank=False)
        self.losses = []

        self.qnet_offline = QNet(self.NUM_CHANNELS, self.GRID_SIZE)
        self.qnet_online = QNet(self.NUM_CHANNELS, self.GRID_SIZE)

        self.env = Environment(self.GRID_SIZE, self.GRID_SIZE)
        self.env.reset()

        self.baseline_trajectory = self.calc_baseline_trajectory()
        self.avg_qs = []

    def calc_baseline_trajectory(self):
        self.env.load_spec_with_index(0)
        x_lst = []

        while True:
            actions = self.env.get_actions()
            action = actions[np.random.randint(len(actions))]
            state, reward, success = self.env.do_action(action)
            x_lst.append(state)
            if not success:
                break

        self.env.reset()
        return x_lst

    def calc_average_q(self):
        cumulative_q = 0
        for x in self.baseline_trajectory:
            max_q = self.qnet_offline.predict_single_max_a(self.env.spec, x)
            cumulative_q += max_q
        return cumulative_q / len(self.baseline_trajectory)

    def update_epsilon(self):
        new_eps = self.epsilon - self.EPS_REDUCTION
        if new_eps >= self.MIN_EPS:
            self.epsilon = new_eps

    def action_to_index(self, action):
        orient, slot = action
        num_slots = 9
        return num_slots * orient + slot

    def action_index_to_tuple(self, a_idx):
        num_slots = 9
        return [a_idx // num_slots, a_idx % num_slots]

    def get_action_with_fitness(self, tetris_state):
        """
        Pick an action based on the heuristic function in the gold
        standard paper.
        """
        def count_holes(field, tops):
            # TODO:
            pass
        temp_env = tetris.TetrisEnv()
        temp_env.set_state(tetris_state)
        actions = temp_env.get_actions()
        action_scores = []
        for action in actions:
            orient, slot = action
            # do action
            next_state, reward, done, _ = temp_env.step(action)
            # measure features
            heights = [next_state.top[col] for col in range(10)]
            heights_offset = heights[1:] + heights[len(heights) - 1]
            agg_height = sum(heights)
            bumpiness = sum([abs(a - b) for a, b in zip(heights, heights_offset)])
            lines_cleared = temp_env.cleared_current_turn
            score = -0.51 * agg_height + 0.76 * lines_cleared + -0.18 * bumpiness
            action_scores.append(score)
            temp_env.set_state(tetris_state)

        best_action_idx = np.argmax(action_scores)
        best_action = actions[best_action_idx]
        return best_action

    def get_action_from_policy(self, tetris_state, epsilon):
        """
        Select a random action with probability epsilon, otherwise
        do argmax_a Q(h(x), a).
        """
        legal_actions = self.env.get_actions()
        random_action = legal_actions[np.random.randint(len(legal_actions))]
        if np.random.random() < epsilon:
            # TODO: 3-way balance between heuristic, random, Q
            # return random_a_idx
            # return self.get_action_with_fitness(tetris_state)
            return random_action
        argmax_a_idx = self.qnet_offline.\
                        predict_single_argmax_a(tetris_state)
        return argmax_a_idx

    def calc_replay_minibatch(self):
        """
        Returns (list(xp, ...), list(a, ...), list(t(y), ...))
        """
        # experiences = self.replay_buffer.weighted_sample(self.MINIBATCH_SIZE,\
        #                     self.sample_weights)
        experiences = self.replay_buffer.random_sample(self.MINIBATCH_SIZE)
        xp_lst = []
        a_lst = []
        y_lst = []
        for exp in experiences:
            spec, x, a, r, xp, terminal = exp
            xp_lst.append(xp)
            a_lst.append(a)
            if terminal:
                y_lst.append(r)
            else:
                max_a_q_pred = self.qnet_offline.\
                        predict_single_max_a(spec, xp)
                y_lst.append(r + self.GAMMA * max_a_q_pred)
        return (xp_lst, a_lst, y_lst)

    def update_offline_network(self):
        self.qnet_offline = copy.deepcopy(self.qnet_online)

    def do_turn(self):
        """
        Inner loop of DQRL algorithm.
        """
        # Record and preprocess current state
        x_old = self.env.canvas.copy()

        # Pick an action via policy
        action = self.get_action_from_policy(x_old, self.epsilon)

        # Execute and observe
        x, reward, success = self.env.do_action(action)

        # Store experience (s*, x, a_idx, r, x', terminal)
        # This is messy since we need to process histories first
        # We only store h in the buffer, x is only available in this loop
        terminal = not success
        exp = (self.env.spec.copy(), x_old, action, reward, x, terminal)
        self.replay_buffer.push(exp)
        # TODO: prioritized sweeping for real
        # weight = reward if reward > 0 else 0.01
        # self.sample_weights.push(weight)

        # Sample minibatch, determine y with offline, update online
        if len(self.replay_buffer) >= self.MINIBATCH_SIZE:
            lst_xp, lst_a, lst_y = self.calc_replay_minibatch()
            print(lst_y)
            self.qnet_online.update(lst_xp, lst_a, lst_y)
            self.update_epsilon()
        return reward

    def train(self):
        game_lengths = []
        crs = []
        avg_qs = []
        # try:
        for ep_num in range(self.NUM_EPISODES):
            self.env.reset()
            self.env.load_spec_with_index(ep_num)
            cumulative_reward = 0
            while not self.env.completed:
                turn_reward = self.do_turn()
                cumulative_reward += turn_reward
            self.frames_since_update += self.env.actions_done
            if self.frames_since_update > self.OFFLINE_UPDATE_FREQ:
                self.update_offline_network()
                self.frames_since_update -= self.OFFLINE_UPDATE_FREQ
                print('\tUpdated network.')
            if ep_num > 0 and ep_num % 10 == 0:
                game_lengths.append(self.env.actions_done)
                crs.append(cumulative_reward)
                avg_q = self.calc_average_q()
                avg_qs.append(avg_q)
                print(f'Episode {ep_num}, eps: {self.epsilon}')
                print(f'\tCR: {cumulative_reward}, turns: {self.env.state.turn}, AvgQ: {avg_q}')
        # except KeyboardInterrupt:
        #     print('Ending early.')
        # except Exception as e:
        #     print(e)
        # finally:
        return game_lengths, crs, avg_qs

    def save_qnet_dict(self):
        torch.save(self.qnet_offline.state_dict(), 'qnet.pt')

    def stack_field_and_piece(self, state):
        def piece_id_to_array(piece_id):
            if piece_id == 0: # O
                return np.array([[1, 1], [1, 1]])
            if piece_id == 1: # I
                return np.array([[1, 1, 1, 1]])
            if piece_id == 2: # L
                return np.array([[1, 0], [1, 0], [1, 1]])
            if piece_id == 3: # J
                return np.array([[0, 1], [0, 1], [1, 1]])
            if piece_id == 4: # T
                return np.array([[0, 1, 0], [1, 1, 1]])
            if piece_id == 5: # S
                return np.array([[0, 1, 1], [1, 1, 0]])
            if piece_id == 6: # Z
                return np.array([[1, 1, 0], [0, 1, 1]])
            else:
                raise Exception()

        field = self.binarize_field(state.field)
        h_field, w_field = field.shape
        piece_array = piece_id_to_array(state.next_piece)
        field_piece_array = np.pad(field, ((0, 0), (0, 4)), 'constant')
        idx_y_piece_start = 0
        idx_x_piece_start = w_field
        idx_y_piece_end = piece_array.shape[0] + idx_y_piece_start
        idx_x_piece_end = piece_array.shape[1] + idx_x_piece_start
        field_piece_array[idx_y_piece_start:idx_y_piece_end,
                          idx_x_piece_start:idx_x_piece_end] = piece_array
        return torch.from_numpy(field_piece_array).reshape((21, 14))\
                .type(torch.FloatTensor)

    def binarize_field(self, field):
        return np.where(field > 0, 1, 0)

if __name__ == '__main__':
    player = Player()
    game_lengths, crs, avg_qs = player.train()
    player.save_qnet_dict()
    with open('game_lengths.csv', 'w') as f:
        for item in game_lengths:
            f.write("%s," % item)
    with open('cumulative_rewards.csv', 'w') as f:
        for item in crs:
            f.write("%s," % item)
    with open('average_qs.csv', 'w') as f:
        for item in avg_qs:
            f.write("%s," % item)

