import numpy as np
import matplotlib.pyplot as plt

id = 4
action = np.genfromtxt(f'neuron_record/16neuron_action_{id}.csv', delimiter=',')
activation = np.genfromtxt(f'neuron_record/16neuron_activation_{id}.csv', delimiter=',')
acts, counts = np.unique(action, return_counts=True)

collecter = [[], [], [], []]
for i, item in enumerate(activation):
    collecter[int(action[i])].append(item)
for i in range(len(collecter)):
    if collecter[i] == []:
        collecter[i] = np.ones([1,16])*(-np.inf)
matrix1 = np.mean(collecter[0], axis=-2)
matrix2 = np.mean(collecter[1], axis=-2)
matrix3 = np.mean(collecter[2], axis=-2)
matrix4 = np.mean(collecter[3], axis=-2)

mat = np.array([matrix1, matrix2, matrix3, matrix4])
np.savetxt('16act.csv', mat, delimiter=',')
mat.transpose([1, 0])
sorted_indices = np.lexsort((mat[0, :], mat[1, :], mat[2, :], mat[3, :]))
sorted_arr = mat.transpose([1,0])[sorted_indices].transpose([1,0])

# indice = np.argsort(-matrix4)
# matrix1 = matrix1[indice].reshape([8,8])
# matrix2 = matrix2[indice].reshape([8,8])
# matrix3 = matrix3[indice].reshape([8,8])
# matrix4 = matrix4[indice].reshape([8,8])

# 创建子图布局
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,2))

# 绘制热力图
color = 'viridis'

heatmap1 = axes.imshow(sorted_arr[2:], cmap=color)
# axes.set_title('NOOP')

# heatmap2 = axes[1].imshow(matrix2, cmap=color)
# axes[1].set_title('FIRE')

# heatmap3 = axes[2].imshow(matrix3, cmap=color)
# axes[2].set_title('RIGHT')

# heatmap4 = axes[3].imshow(matrix4, cmap=color)
# axes[3].set_title('LEFT')

# 添加颜色条
# cbar = fig.colorbar(heatmap1, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.6)

# 设置整体图形标题
fig.suptitle('Average Neuron Activation')

# 调整子图之间的间距
# plt.tight_layout()

# 显示图形
plt.savefig(f'img/activation_pattern_{id}.png', dpi=200)
plt.show()
