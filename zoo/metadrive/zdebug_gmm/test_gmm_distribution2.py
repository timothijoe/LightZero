import torch
from torch.distributions import MixtureSameFamily, MultivariateNormal
import numpy as np
# 构建GMM的参数
weights = torch.tensor([0.3, 0.5, 0.2],requires_grad=True,dtype=float)  # 组件的权重
means = torch.tensor([[1.0, 1.0], [-1.0, 2.0], [3.0, 3.0]],requires_grad=True,dtype=float)  # 组件的均值
covs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],requires_grad=True,dtype=float)  # 组件的协方差

# weights = weights.double()
# means = means.double()
# covs = covs.double()
# 构建GMM的分量分布
component_distributions = MultivariateNormal(means, covs)

# 构建MixtureSameFamily对象
gmm = MixtureSameFamily(torch.distributions.Categorical(weights), component_distributions)

# 生成输入数据
input_data = torch.tensor([[0.2, 0.3], [1.5, 1.7], [0.8, 0.9]])
sample = np.array([[1.90716212, 3.3556789 ],
 [1.11701485, 2.82362575],
 [1.08452658, 3.70330748]])
input_data = torch.from_numpy(sample)
# 计算log_prob
log_probs = gmm.log_prob(input_data)

# 采样
samples = gmm.sample((100,))

print("Log Probabilities:")
print(log_probs)
# print("\nSamples:")
# print(samples)
log_probs.sum().backward()
print("d_mu = {}".format(means.grad))
print("d_sigma = {}".format(covs.grad))
print("d_weights = {}".format(weights.grad))