import argparse
import time
import os
from modqn import *
from env import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        usage="To run a molecular generation/optimization process, parameter '--output_path' is required.\n"
              "If no initial molecule is given (parameter '--init_smi' is None), it will be a molecular generation process starting from CC.\n"
              "Using parameter '-h' to see more optional parameters.")
    parser.add_argument('--init_smi', metavar='default=CC', type=str, default='CC')
    parser.add_argument('--program_name', metavar='default=cc', type=str, default='cc')
    # parser.add_argument('--output_path', metavar='required', type=str, default=r'.')
    parser.add_argument('--output_path', metavar='required', type=str, required=True)
    parser.add_argument('--pocket_file', metavar='binding site file (.pdb)', type=str, default=r'.\case1\nilotinib_pocket6A.pdb')
    parser.add_argument('--lr', metavar='default=0.01', type=float, default=0.01)
    parser.add_argument('--lr_decrease', metavar='default=0.1 (per 200 episodes), multiplies, from lr to 0.0001', type=float, default=0.1)
    parser.add_argument('--batch_size', metavar='default=64', type=int, default=64)
    parser.add_argument('--gamma', metavar='default=1.0', type=float, default=1.0)
    parser.add_argument('--learning_iter', metavar='default=1', type=int, default=1)
    parser.add_argument('--target_iter', metavar='default=10', type=int, default=10)
    parser.add_argument('--memory_capacity', metavar='default=1000', type=int, default=1000)
    parser.add_argument('--max_episode', metavar='default=4000', type=int, default=4000)
    parser.add_argument('--max_step', metavar='default=5', type=int, default=5)
    parser.add_argument('--epsilon', metavar='default=0.95', type=float, default=0.95)
    parser.add_argument('--epsilon_increase', metavar='default=0.05 (per 50 episodes), adds, from 0.5 to epsilon', type=float, default=0.05)
    return parser.parse_args()


args = parse_args()

if __name__ == "__main__":
    log_file = open(args.output_path + os.sep + 'log_' + args.program_name + '.txt', 'w')
    log_file.write(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime()))
    log_file.write('Parameters:\n')
    for i in vars(args):
        log_file.write(i + ': ' + str(vars(args)[i]) + '\n')

    env = main_env(reward_list=['SAscore', 'QED', 'Affinity'],
                   init_smi=args.init_smi,
                   pocket_file_path=args.pocket_file)
    model = MODQN(reward_list=env.reward_list,
                  lr=args.lr,
                  lr_decrease=args.lr_decrease,
                  target_iter=args.target_iter,
                  state_dim=1024,
                  # molecular descriptors in CSstep: MorganFingerprint(mol, radius=3, nBits=1024)
                  action_dim=1024,
                  # in the MDP-based molecular modification proposed by Google, the action is equivalent to the next state
                  gamma=args.gamma,
                  epsilon=args.epsilon,
                  epsilon_increase=args.epsilon_increase,
                  double_q=True)
    M = Memory(args.memory_capacity,
               state_dim=1024,
               action_dim=1024,
               reward_dim=model.reward_dim)
    M_smi = Memory_smiles(args.memory_capacity)

    log_file.write('Objective_properties: {}\n'.format(env.reward_list))
    log_file.write('\nEpisode,{},SMILES\n'.format(','.join(env.reward_list)))

    for episode in range(args.max_episode):
        if episode != 0:
            if episode % 200 == 0:
                model.lr_down()
            if episode % 50 == 0:
                model.epsilon_up()
            if episode % 1000 == 0:
                # multi-stage fine-tuning strategy
                cleaned_affinity_reward = np.zeros(len(M.data), dtype=float)
                # search for the best molecule generated in the last (memory_capacity/max_step) epsidoes of the current stage
                for i in range(len(M.data)):
                    if M.data[i, -len(env.reward_list)-1] == (args.max_step-1) and M.data[i, -3] > 0.810 and M.data[i, -2] > 0.266:
                        # memory: state(1024), action(1024), step_index(1), SAscore(1), QED(1), Affinity(1), see env.py
                        # Nilotinib SAscore: 0.810, QED: 0.266
                        cleaned_affinity_reward[i] = M.data[i, -1]
                if np.any(cleaned_affinity_reward != 0.0):
                    best_row_index = np.argmax(cleaned_affinity_reward)
                    env.init_smi = M_smi.data[best_row_index]
                    print('The initial molecule has been reset to %s' % env.init_smi)
                    print('Reward: {}, {}, {}'.format(round(M.data[best_row_index, -3], 3),
                                                      round(M.data[best_row_index, -2], 3),
                                                      round(M.data[best_row_index, -1]*10, 3)))
                    # the affinity reward has been reduced to 1/10, see env.py
                    with open(args.output_path + os.sep + 'initial_mols_%s.txt' % args.program_name, 'a') as f:
                        f.write('{},{},{},{},{}\n'.format(str(episode),
                                                          round(M.data[best_row_index, -3], 3),
                                                          round(M.data[best_row_index, -2], 3),
                                                          round(M.data[best_row_index, -1]*10, 3),
                                                          env.init_smi))
                    M.reset()
                    M_smi.reset()
                    # clear memory
                    model.lr = args.lr
                    model.epsilon = 0.5
                    # reset the learning rate and random exploration rate
                else:
                    print('There is no molecule meeting the update criteria in the memory library! Continue to train for 1000 episodes...')
                    with open(args.output_path + os.sep + 'initial_mols_%s.txt' % args.program_name, 'r') as f:
                        lines = f.readlines()
                    last_line = lines[-1]
                    with open(args.output_path + os.sep + 'initial_mols_%s.txt' % args.program_name, 'a') as f:
                        f.write(str(episode) + last_line[last_line.index(','):])
        else:
            with open(args.output_path + os.sep + 'initial_mols_%s.txt' % args.program_name, 'w') as f:
                f.write('{},{},{},{},{}\n'.format('0', cal_norm_SAscore(args.init_smi),
                                                  cal_QED(args.init_smi),
                                                  predict_Affinity(env.model, env.pocket, args.init_smi),
                                                  args.init_smi))

        state = env.reset()
        for step in range(args.max_step):
            action, state_fps, action_fps = model.choose_action(state, step)
            state_ = env.step(action)
            reward = env.reward(args.max_step)
            M.store(state_fps, action_fps, step, reward)
            M_smi.store(action)
            if M.pointer > args.memory_capacity and (step+1) % args.learning_iter == 0:
                data, indexes = M.sample(args.batch_size)
                data_smi = M_smi.sample(indexes)
                model.learn(data, data_smi, args.max_step)
            state = state_

            if step == args.max_step - 1:
                print('{},{},{},{},{}'.format(episode, reward[0], reward[1], round(reward[2]*10, 3), action))
                log_file.write('{},{},{},{},{}\n'.format(episode, reward[0], reward[1], round(reward[2]*10, 3), action))

    log_file.write(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime()))
    log_file.close()

    df = pd.read_csv(args.output_path + os.sep + 'log_' + args.program_name + '.txt', header=18)
    df = df[:-1]
    if len(df) % 100 != 0:
        episode_list = [100 * (i + 1) for i in range(len(df) // 100 + 1)]
    else:
        episode_list = [100 * (i + 1) for i in range(len(df) // 100)]
    median_SAscore = []
    median_QED = []
    median_Affinity = []
    for i in range(0, len(df), 100):
        sub_df = df.iloc[i:i + 100]
        median_SAscore.append(round(sub_df['SAscore'].median(), 4))
        median_QED.append(round(sub_df['QED'].median(), 4))
        median_Affinity.append(round(sub_df['Affinity'].median(), 4))

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax1 = plt.subplots(figsize=(6, 6), dpi=150)
    ax1.plot(episode_list, median_SAscore, label='Normized_SAscore', color='black')
    ax1.plot(episode_list, median_QED, label='QED', color='red')
    ax1.set_xlabel('Episode', fontsize=16)
    ax1.set_xticks(np.arange(0, len(df) + 1, 1000))
    ax1.set_ylabel('Median of SAscore and QED per 100 episodes', fontsize=16)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.tick_params(direction='in', labelsize=16)
    ax2 = ax1.twinx()
    ax2.plot(episode_list, median_Affinity, label='-lg(Affinity)', color='blue')
    ax2.set_ylabel('Median of -lg(Affinity) per 100 episodes', fontsize=16)
    ax2.set_yticks(np.arange(0.0, 11.0, 1.0))
    ax2.tick_params(direction='in', labelsize=16)
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    ax1.legend(handles, labels, fontsize=12, loc='best', framealpha=0.8, ncol=1)
    plt.savefig(args.output_path + os.sep + 'training_' + args.program_name + '.svg', format='svg')
    plt.show()
