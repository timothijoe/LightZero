import torch
from torch.distributions import MixtureSameFamily, MultivariateNormal, Normal, Independent, Categorical
import numpy as np

# 设置batch大小
batch_size = 4

# 构建GMM的参数
weights = torch.ones(batch_size, 3) / 3  # (batch_size, num_components)
weights.requires_grad = True
means = torch.randn(batch_size, 3, 2)  # (batch_size, num_components, event_shape)
means.requires_grad = True
stddevs = torch.ones(batch_size, 3, 2)  # (batch_size, num_components, event_shape)
stddevs.requires_grad = True

# 构建GMM的分量分布
component_distributions = Independent(Normal(means, stddevs), 1)
# 构建MixtureSameFamily对象
gmm = MixtureSameFamily(Categorical(weights), component_distributions)

# 生成输入数据
num_samples = 3
input_data = torch.randn((num_samples, batch_size, 2))  # (num_samples, batch_size, event_shape)

# 计算log_prob
log_probs = gmm.log_prob(input_data)

# 采样
samples = gmm.sample((100,))

print("Log Probabilities:")
print(log_probs)

# Backward pass
log_probs.sum().backward()

print("d_mu = {}".format(means.grad))
print("d_sigma = {}".format(stddevs.grad))
print("d_weights = {}".format(weights.grad))