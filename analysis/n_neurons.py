import matplotlib.pyplot as plt
import numpy as np
import os
import re

log_dir = '/Users/annihillusion/Workflow/ncps_research/data_for_draw/8&12_50epoch'

def extract_record(log_folder, file_name):
    file_path = os.path.join(log_folder, file_name)
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    reward_list = []
    # 打开文本文件并逐行读取内容
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取train_loss和reward
            match = re.search(r'train_loss=([\d.]+), val_loss=([\d.]+), val_acc=([\d.]+)', line)
            if match:
                train_loss_list.append(float(match.group(1)))
                valid_loss_list.append(float(match.group(2)))
                valid_acc_list.append(float(match.group(3)))
            match = re.search(r'Mean return ([\d.]+)', line)
            if match:
                reward_list.append(float(match.group(1)))
    return np.array(train_loss_list), np.array(valid_loss_list), np.array(valid_acc_list), np.array(reward_list)

def sub_draw(num):
    global axes
    global log_dir
    
    train_loss, val_loss, val_acc, reward = extract_record(log_dir, f'{num}neurons_ncp.txt')
    axes[0].plot(train_loss, label=f'{num} neurons')
    axes[1].plot(val_loss, label=f'{num} neurons')
    axes[2].plot(val_acc, label=f'{num} neurons')
    axes[3].plot(reward, label=f'{num} neurons')

    axes[0].set_title('Train loss')
    axes[1].set_title('Valid loss')
    axes[2].set_title('Valid acc')
    axes[3].set_title('Reward')
    # axes[i, j].set_xlabel('Epoch')
    # axes[i, j].set_ylabel('Loss')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()


fig, axes = plt.subplots(1, 4, figsize=[16,4], sharex=True)

sub_draw(8)
sub_draw(12)
# sub_draw(16)
# sub_draw(20)
# sub_draw(24)
# sub_draw(28)
sub_draw(32)

plt.savefig('img/8&12_50epoch.png', dpi=200)
plt.show()