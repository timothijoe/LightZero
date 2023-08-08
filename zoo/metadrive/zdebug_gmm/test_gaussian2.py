import torch
import numpy as np
from scipy.stats import multivariate_normal

def gaussian_mixture_pdf(weight, mean, covariance, sample):
    pdf = np.zeros(sample.shape[0])
    for i in range(len(weight)):
        pdf += weight[i] * multivariate_normal.pdf(sample, mean=mean[i], cov=covariance[i])
    return pdf
# 定义高斯混合模型参数
weight = [0.3, 0.5, 0.2]
mean = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
covariance = torch.tensor([[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[3, 2], [2, 3]]], dtype=torch.float)

mean = torch.tensor([[1, 1], [-1, 2], [3, 3]], dtype=torch.float)
covariance = torch.tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=torch.float)

# 生成样本数据
#sample = np.random.normal(loc=[1.5, 3.5], scale=0.5, size=(100, 2))
sample = np.random.normal(loc=[1.5, 3.5], scale=0.5, size=(3, 2))
# print(sample)
sample = np.array([[1.90716212, 3.3556789 ],
 [1.11701485, 2.82362575],
 [1.08452658, 3.70330748]])

print(sample)
# 计算概率密度函数
pdf = gaussian_mixture_pdf(weight=weight, mean=mean, covariance=covariance, sample=sample)
print(pdf)
print('log pdf')
print(np.log(pdf))