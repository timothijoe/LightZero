import torch
from torch.distributions import MixtureSameFamily, MultivariateNormal

# 构建GMM的参数
weights = torch.tensor([1.0])  # 组件的权重
means = torch.tensor([[0.0, 0.0]])  # 组件的均值
covs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # 组件的协方差

# 构建GMM的分量分布
component_distributions = MultivariateNormal(means, covs)

# 构建MixtureSameFamily对象
gmm = MixtureSameFamily(torch.distributions.Categorical(weights), component_distributions)

# 生成输入数据
input_data = torch.tensor([[0.2, 0.3], [1.5, 1.7], [0.8, 0.9]])

# 计算log_prob
log_probs = gmm.log_prob(input_data)

# 采样
samples = gmm.sample((100,))

print("Log Probabilities:")
print(log_probs)
# print("\nSamples:")
# print(samples)
