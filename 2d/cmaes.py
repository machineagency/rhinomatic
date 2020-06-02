import numpy as np
from torch import save, load, from_numpy, FloatTensor
from scipy.linalg import sqrtm
from sklearn import preprocessing
import imageio
import vae
import canvas

class Featurizer:
    def __init__(self, do_training=False):
        self.state_dict_filepath = 'vae_state_dict.pt'
        self.vae = vae.VAE()
        self.EPOCHS = 10
        if do_training:
            self.train_vae()
        else:
            self.load_vae_state_dict()

    def train_vae(self):
        for epoch in range(self.EPOCHS):
            self.vae.train_model(epoch)
            self.vae.test(epoch)
        save(self.vae.state_dict(), self.state_dict_filepath)

    def load_vae_state_dict(self):
        self.vae.load_state_dict(load(self.state_dict_filepath))
        self.vae.eval()

    def featurize(self, img):
        img_single_batch = from_numpy(img).reshape((1, 3, 32, 32))\
                            .type(FloatTensor)
        _, mu, logvar = self.vae(img_single_batch)
        # what do i do with variance
        return mu

class CMAES:
    def __init__(self, train_featurizer=False):
        self.featurizer = Featurizer(train_featurizer)
        self.canvas = canvas.Canvas(32, 32)
        self.MAX_ITER = 5
        self.FITNESS_THRESH = 10000
        self.NUM_EVAL_GAMES = 10
        self.dim = 20
        self.init_weightset = np.zeros(self.dim)

        # Calculate hyperparameters from dimension
        self.lam = int(4 + np.floor(3 * np.log(self.dim)))
        self.mu = int(np.floor(self.lam / 2))
        self.w = np.zeros(self.mu)
        for i in range(1, self.mu + 1):
            self.w[i - 1] = (np.log(self.mu + 1) - np.log(i)) / (self.mu\
                    * np.log(self.mu + 1)\
                    - np.sum([np.log(j) for j in range(1, self.mu + 1)]))
        self.mu_eff = 1 / (self.w @ self.w)
        self.c_sig = (self.mu_eff + 2) / (self.dim + self.mu_eff + 3)
        self.d_sig = 1 + 2 * np.max([0.0,\
                np.sqrt((self.mu_eff - 1)\
                / (self.dim + 1)) - 1])\
                + self.c_sig
        self.c_c = 4 / (self.dim + 4)
        self.mu_co = self.mu_eff
        self.c_co = (1 / self.mu_co) * (2 / (self.dim + np.sqrt(2)) ** 2)\
                + (1 - 1 / self.mu_co) * np.min([1.0, (2 * self.mu_eff - 1)\
                / ((self.dim + 2) ** 2 + self.mu_eff)])
        self.gauss_dist_norm = np.sqrt(self.dim) * (1 - (1 / 4 * self.dim)\
                + (1 / (21 * self.dim ** 2)))
        pass

    def sample_dist(self, mean, cov_m):
        return np.random.multivariate_normal(mean, cov_m)

    def evaluate_single(self, x, gameplay_seeds):
        """
        Given a (1 x DIM) weightset for the tetris value
        function, run evaluation on each to determine fitness.
        NOTE: this must return _higher_ scores for fitter samples.
        """
        assert x.shape == (self.dim,) or x.shape == (self.dim, 1)
        avg_score = 0
        avg_turns = 0
        for game_num in range(self.NUM_EVAL_GAMES):
            score, turns = self.play_tetris_with_weightset(x)
            np.random.seed()
            avg_score += score
            avg_turns += turns
        return (avg_score / self.NUM_EVAL_GAMES, avg_turns / self.NUM_EVAL_GAMES)

    def test_fitness_fn(self, x):
        """
        Should have a global minimum at (20, ...)
        """
        x_off = x - 20
        poly = -(x_off ** 2)
        actual_sum = np.sum(poly)
        return -abs(actual_sum)

    def play_cad_with_weightset(self, weightset):
        def q(state, action, spec):
            next_state = self.canvas.peek_action(action)
            q_stack = np.stack([state, next_state, spec])
            features = self.featurizer.featurize(q_stack)
            features = features.reshape((3 * self.dim, 1)).detach().numpy()
            return np.dot(weightset, features)[0]

        filepath = './test/drawings'
        test_set_size = 32
        images = [imageio.imread(f'{filepath}/{i}.png')\
                    for i in range(test_set_size)]
        images = [image.reshape(32, 32, 3) for image in images]
        specs = [image[:, :, 0] for image in images]

        total_ioc = 0
        for spec_idx, spec in enumerate(specs):
            print(f'On spec {spec_idx}')
            self.canvas.clear_canvas()
            self.canvas.clear_primitives()
            while True:
               orig_state = self.canvas.canvas.copy()
               actions = self.canvas.get_reasonable_actions(spec)
               print(f'\tConsidering {len(actions)} actions...')
               q_vals = [q(orig_state, a, spec) for a in actions]
               best_action = actions[np.argmax(q_vals)]
               print(f'... did {best_action}.')
               action_succeeded = self.canvas.do_action(best_action)
               if not action_succeeded:
                   break
            total_ioc += self.canvas.intersection_over_union(spec)
        return total_ioc / test_set_size

    def play_tetris_with_weightset(self, weightset):
        def v(state):
            featurizer = Featurizer()
            features = featurizer.featurize(state)
            return np.dot(weightset, features)

        env = tetris.TetrisEnv()
        env.reset()
        total_reward = 0
        while True:
            # Preview next states
            orig_state = env.state.copy()
            actions = env.get_actions()
            peek_next_states = []
            for action in actions:
                peek_next_state, _, _, _ = env.step(action)
                peek_next_states.append(peek_next_state)
                env.set_state(orig_state)

            # Find best action and do for real
            values = [v(s) for s in peek_next_states]
            best_action = actions[np.argmax(values)]
            next_state, reward, done, _ = env.step(best_action)
            total_reward += reward
            if done:
                break
        return (total_reward, next_state.turn)

    def sort_by_fitness(self, x, f):
        """
        Given X which contains sample columns of length DIM each, sort
        the columns such that the 0th scorest best by fitness, descending
        thereafter.
        F contains a scalar fitness score for each column.
        Higher fitnesses are better.
        """
        # Sorts f with highest fitness at index 0
        sort_indices = f.argsort()[::-1]
        return x[sort_indices]

    def inv_sqrt(self, m):
        """
        Computes inverse square root of a matrix M.
        """
        return np.linalg.inv(sqrtm(m))

    def train(self, starting_mean=None):
        # Things we will update
        if starting_mean is None:
            mean = np.random.rand(self.dim)
        else:
            mean = starting_mean
        sigma = 1
        C = np.identity(self.dim)
        p_sig = np.zeros(self.dim)
        pc = np.zeros(self.dim)

        # Store stats
        running_means = []
        running_fitnesses = []
        running_turns = []

        # X contains LAM sample, each with SELF.DIM features
        z = np.zeros((self.lam, self.dim))
        x = np.zeros((self.lam, self.dim))
        f = np.zeros(self.lam)
        # turns is for scorekeeping only
        turns = np.zeros(self.lam)
        for t in range(self.MAX_ITER):
            print(f'CMAES iter {t}')
            # Draw population samples
            gameplay_seeds = np.random.randint(0, 100, self.NUM_EVAL_GAMES)
            gameplay_seeds += self.true_random[t]
            for i in range(self.lam):
                print(f'\tDraw / eval sample {i} ...')
                unnorm_z = self.sample_dist(np.zeros(self.dim), C)
                z[i] = preprocessing.normalize(unnorm_z.reshape(1, -1))
                x[i] = mean + sigma * z[i]
                f[i], turns[i] = self.evaluate_single(x[i], gameplay_seeds)
                print(f'\t... scored {f[i]}, turns {turns[i]}')
            x = self.sort_by_fitness(x, f)
            mean_old = mean.copy()
            mean = np.sum([self.w[i] * x[i] for i in range(self.mu)], axis=0)
            print(f'Best F this round: {np.max(f)}')
            print(f'Best turns this round: {np.max(turns)}')
            print(f'New mean: {mean}')
            print(f'Sigma: {sigma}')
            running_means.append(mean)
            running_fitnesses.append(np.max(f))
            running_turns.append(np.max(turns))

            # Update global step
            p_sig = (1 - self.c_sig) * p_sig\
                + np.sqrt(self.c_sig * (2 - self.c_sig) * self.mu_eff)\
                * self.inv_sqrt(C) @ ((mean - mean_old) / sigma)
            sigma = sigma * np.exp((self.c_sig / self.d_sig)\
                    * ((np.linalg.norm(p_sig) / self.gauss_dist_norm) - 1))

            # Update covariance
            pc_indic = 1 if np.linalg.norm(p_sig) < 1.5 * np.sqrt(self.dim) else 0
            pc = (1 - self.c_c) * pc + pc_indic\
                * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_co)\
                * ((mean - mean_old) / sigma)
            mu_update = np.sum([self.w[i] * z[i].reshape((self.dim, 1))\
                    @ z[i].reshape((1, self.dim)) for i in range(self.mu)], 0)
            C = (1 - self.c_co) * C + (self.c_co / self.mu_co)\
                    * (pc.reshape((self.dim, 1)) @ pc.reshape((1, self.dim)))\
                    + self.c_co * (1 - (1 / self.mu_co)) * mu_update

            if abs(f[0]) >= self.FITNESS_THRESH:
                break

        with open('running_means.csv', 'w') as f:
            for item in running_means:
                f.write("%s," % item)
        with open('fitnesses.csv', 'w') as f:
            for item in running_fitnesses:
                f.write("%s," % item)
        with open('turns.csv', 'w') as f:
            for item in running_turns:
                f.write("%s," % item)
        return mean

    def test_play(self):
        weightset = np.array([-1, 4, -1, -1, -1, 1, -1, -1])
        total_reward = self.play_tetris_with_weightset(weightset)
        return total_reward

    def run_final_weights(self):
        cumulative_reward = 0
        num_games = 20
        final_result_seed = 801382
        np.random.seed(final_result_seed)
        for game_num in range(num_games):
            print(f'Playing game # {game_num}...')
            total_reward, turn = self.play_tetris_with_weightset(\
                    self.final_weightset)
            cumulative_reward += total_reward
            print(f'... score: {total_reward}, turns lasted: {turn}')
        np.random.seed()
        return cumulative_reward / num_games

    def test(self):
        score = self.play_cad_with_weightset(np.ones(3 * self.dim))
        print(score)
        # print('Sanity check fitness function')
        # print(self.test_fitness_fn(np.array([20, 20])))
        # print(self.test_fitness_fn(np.array([13, 12])))
        # print('Let\'s play a game')
        # print(self.test_play())
        # mean = self.train(self.init_weightset)
        # print('Training result')
        # print(mean)
        # final_result = self.run_final_weights()
        # print(f'Final average score: {final_result}')

