import matplotlib.pyplot as plt
import numpy as np
import os
import re

log_dir = 'D:\\ncps_research\\data_for_draw\\noRecur'

def extract_record(log_folder, file_name):
    file_path = os.path.join(log_folder, file_name)
    train_loss_list = []
    reward_list = []
    # 打开文本文件并逐行读取内容
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取train_loss和reward
            match = re.search(r'train_loss=([\d.]+)', line)
            if match:
                train_loss = float(match.group(1))
                train_loss_list.append(train_loss)
            match = re.search(r'Mean return ([\d.]+)', line)
            if match:
                reward = float(match.group(1))
                reward_list.append(reward)
    return np.array(train_loss_list), np.array(reward_list)

def sub_draw(i, j, num):
    global axes
    global log_dir
    
    loss1, reward1 = extract_record(log_dir, f'{num}neurons_ncp.txt')
    loss2, reward2 = extract_record(log_dir, f'{num}neurons_ncp_noRecur.txt')
    axes[i, j].plot(loss1, label='ncp', color='g')
    axes[i, j].plot(loss2, label='noRecur', color='b')
    axess = axes[i, j].twinx()
    axess.plot(reward1, label='ncp', color='g')
    axess.plot(reward2, label='noRecur', color='b')
    axes[i, j].set_title(f'{num} neurons')
    # axes[i, j].set_xlabel('Epoch')
    # axes[i, j].set_ylabel('Loss')
    axes[i, j].legend()

fig, axes = plt.subplots(2, 3, figsize=[12,8], sharey=True)
sub_draw(0, 0, 8)
sub_draw(0, 1, 12)
sub_draw(0, 2, 16)
sub_draw(1, 0, 20)
sub_draw(1, 1, 24)
sub_draw(1, 2, 28)

plt.savefig('../img/noRecur.png', dpi=200)
plt.show()
