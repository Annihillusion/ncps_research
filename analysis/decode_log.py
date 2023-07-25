import numpy as np
import os
import re
import matplotlib.pyplot as plt

log_folder = '/Users/annihillusion/Workflow/log'
collector = []
train_loss_list = []
val_loss_list = []
reward_list = []
# 遍历log文件夹中的所有文本文件
for file_name in os.listdir(log_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(log_folder, file_name)
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
    collector.append(np.array([train_loss_list[0:25], reward_list[0:25]]))
    train_loss_list.clear()
    reward_list.clear()
loss = np.array(collector)

epochs = range(1, loss.shape[-1]+1)
# 绘制loss曲线
for item in loss:
    plt.plot(epochs, item[1])

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 显示图形
plt.show()