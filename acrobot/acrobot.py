import numpy as np
import matplotlib.pyplot as plot
import gym
import pickle
import sys

# defining text colors
CRED = '\033[91m'
CGREEN = '\033[92m'
CYELLOW = '\033[93m'
CEND = '\033[0m'

# defining keys
ESC = 27
ENTER = 13

# importing the environment from gym lib
env = gym.make("Acrobot-v1")
env.reset()


# ================================= #
#           ENV RECON               #
# ================================= #


# print("States: ", env.observation_space)
# print("Actions: ", env.action_space)
# print("State min: ", env.observation_space.low)
# print("State max: ", env.observation_space.high)
# print("State start: ", env.reset())
# print("Step: ", env.step(1))


# ================================= #
#                DEV                #
# ================================= #


def Qlearning(env, alpha, gamma, eps, eps_int, eps_disp, policy, f_policy):
    # memory warning
    key = input("\n=============================================================="
                + CYELLOW + "\nWARNING!"
                + CEND + " This program consumes about 7 GB of memory." + CRED + " Be aware."
                + CEND + "\n==============================================================\n"
                + "\nTo go on, input " + CGREEN + "continue"
                + CEND + ". Any other input will stop the program.\n\tInput: ")

    if key != "continue":
        sys.exit(CRED + "\n{}".format(key) + CEND + " => Stopped by user.")
    else:
        print(CGREEN + "\n{}".format(key) + CEND + " => Running code:\n\n")

    # policy reading state
    pol_read_error = False

    # initializing Q-table
    Q = {}

    # actions are already discrete in this env
    n_actions = [3]
    # print(n_actions)

    # load/generate policy (Q-table)
    if policy:
        try:
            print("Loading policy from {}...".format(f_policy))
            with open(f_policy, "rb") as fr:
                Q = pickle.load(fr)
            print("Policy loaded from file. ({} GB used)".format(round(sys.getsizeof(Q) / 1000000000), 3))
        except(OSError, IOError) as e:
            print("Couldn't load policy. A random policy will be generated.\n\t({})".format(e))
            pol_read_error = True

    # if not using policy or error on loading
    if not policy or pol_read_error:
        # initializing Q-table
        print("Generating random Q-table...")

        # making state components discrete so that our Q-table is not huge
        n_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 10, 10, 10, 1, 1])
        n_states = np.round(n_states, 0).astype(int) + 1
        # print(n_states)

        Q = np.random.uniform(-1, 1, (n_states[0], n_states[1], n_states[2],
                                      n_states[3], n_states[4], n_states[5], n_actions[0]))
        print("Random Q-table alocated. ({} GB used)".format(round(sys.getsizeof(Q) / 1000000000), 3))
        # print(Q)

    # initializing reward related vectors
    rew_list = []
    ave_rew_list = []

    # calculating episolon
    epsilon = 1 - alpha

    # calculating reduction factor per episode
    reduct = epsilon / eps

    # boolean for displaying last 10 episodes
    display = False

    print("Q-table size: {} GB".format(round(sys.getsizeof(Q) / 1000000000), 3))

    print("================================\n"
          "\tTrain Starting\n"
          "================================\nWorst reward possible: -1.0\nBest reward possible: 0.0"
          + "\nRunning {} episodes".format(eps)
           + "\nAlpha (Learning ratio): {}".format(alpha)
           + "\nGamma (Discount ratio): {}\n================================\n".format(gamma) )

    # stating the algorithm
    for i in range(eps):

        # initializing internal parameters
        finished = False
        total_rew = 0
        rew = 0
        cur_state = env.reset()

        # turning current state into discrete values
        d_cur_state = (cur_state - env.observation_space.low) * np.array([10, 10, 10, 10, 1, 1])
        d_cur_state = np.round(d_cur_state, 0).astype(int)

        # stating train
        while not finished:

            # display the last x episodes
            if i >= eps - eps_disp:
                if not display:
                    input("{} episodes left. Press enter for start rendering.".format(eps_disp))
                    display = True
                else:
                    env.render()

            # exploit X explore decision (episolon greedy)
            if np.random.random() < alpha:
                # exploiting case
                action = np.argmax(Q[d_cur_state[0], d_cur_state[1], d_cur_state[2],
                                     d_cur_state[3], d_cur_state[4], d_cur_state[5]])
            else:
                # exploring case
                action = np.random.randint(0, n_actions)

            # take the decided action and receive outputs
            next_state, rew, finished, info = env.step(int(action))

            # turning next state into discrete values
            d_next_state = (next_state - env.observation_space.low) * np.array([10, 10, 10, 10, 1, 1])
            d_next_state = np.round(d_next_state, 0).astype(int)

            # check if reached terminal state
            if finished:
                # define last Q-value
                Q[d_cur_state[0], d_cur_state[1], d_cur_state[2],
                  d_cur_state[3], d_cur_state[4], d_cur_state[5], action] = rew
            else:
                # update Q-value for current state
                delta = alpha * (rew + gamma *
                                 np.max(Q[d_next_state[0], d_next_state[1],
                                          d_next_state[2], d_next_state[3],
                                          d_next_state[4], d_next_state[5]])
                                 - Q[d_cur_state[0], d_cur_state[1], d_cur_state[2],
                                     d_cur_state[3], d_cur_state[4], d_cur_state[5],
                                     action])
                Q[d_cur_state[0], d_cur_state[1], d_cur_state[2],
                  d_cur_state[3], d_cur_state[4], d_cur_state[5], action] += delta

            # update reward related
            total_rew += rew
            d_cur_state = d_next_state

        # update epsilon value
        epsilon -= reduct

        # update reward list
        rew_list.append(rew)

        # print reward for last x episodes
        if i >= eps - eps_disp:
            if rew < 0:
                print("episode: #{} | reward: ".format(i + 1) + CRED + "{}".format(rew) + CEND)
            elif rew == 0:
                print("episode: #{} | reward: ".format(i + 1) + CGREEN + "{}".format(rew) + CEND)

        # print progress each x episodes
        if (i + 1) % eps_int == 0:
            # calculate arithmetic mean of rewards
            ave_rew = np.mean(rew_list)
            # add it to the list of averages
            ave_rew_list.append(ave_rew)
            # clear episode list of rewards
            rew_list = []

            if ave_rew < 0:
                print("episodes: #{} - #{} | average reward: ".format(i - eps_int + 1, i + 1) +
                      CRED + "{}".format(ave_rew) + CEND)
            elif ave_rew == 0:
                print("episodes: #{} - #{} | average reward: ".format(i - eps_int + 1, i + 1) +
                      CGREEN + "{}".format(ave_rew) + CEND)
    env.close()

    # save policy
    if policy:
        print("Saving policy to disk ({} GB). This may take a while, so please be patient.".format(
            round(sys.getsizeof(Q) / 1000000000), 3))
        with open(f_policy, "wb") as fw:
            pickle.dump(Q, fw)
        print("Policy saved as {}".format(f_policy))

    return ave_rew_list


# ================================= #
#               MAIN                #
# ================================= #

# control variables
learning = 0.9
discount = 0.9
episodes = 5000
episodes_interval = 10
episodes_display = 5
use_policy = True
policy_file = "acrobot_good_policy.bin"
graph_file = "acrobot-stats.png"

# run algorithm
rewards = Qlearning(env, learning, discount, episodes, episodes_interval, episodes_display, use_policy, policy_file)

# make graph
plot.plot(episodes_interval * (np.arange(len(rewards)) + 1), rewards)
plot.xlabel('Episodes')
plot.ylabel('Average Reward')
plot.title('Acrobot Q-Learning: Average Reward vs Episodes')
plot.savefig(graph_file)
plot.close()
print("Graph saved as {}".format(graph_file))
