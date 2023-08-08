import torch

# 生成从0到3的数字
sequence = torch.ones(4).unsqueeze(-1)

# 生成从0到20的数字
numbers = torch.arange(21).unsqueeze(0)

# 在第一个维度上重复数字，并在第二个维度上保持顺序
tensor = sequence * numbers

print(tensor)
print(tensor[0])
print(tensor[:,5])
policy_logits = tensor
gmm_num = 3
batch_size = 4
event_shape = 3
weights = policy_logits[:, 0:gmm_num]
means = policy_logits[:, gmm_num:(gmm_num+event_shape*gmm_num)].reshape(batch_size, gmm_num, event_shape)
stds = policy_logits[:, (gmm_num+event_shape*gmm_num):(gmm_num+2*event_shape*gmm_num)].reshape(batch_size, gmm_num, event_shape)

print(weights)
print('mu: {}'.format(means))
print('sigma: {}'.format(stds))