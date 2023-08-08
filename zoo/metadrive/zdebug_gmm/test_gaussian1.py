import numpy as np
from scipy.stats import multivariate_normal

def multivariate_gaussian_pdf(x, mean, cov):
    """
    计算多元高斯分布的概率密度函数
    :param x: 样本值，shape为(n, d)，其中n为样本数量，d为样本维度
    :param mean: 均值，shape为(d,)
    :param cov: 协方差矩阵，shape为(d, d)
    :return: 概率密度函数值，shape为(n,)
    """
    pdf = multivariate_normal(mean=mean, cov=cov)
    return pdf.pdf(x)


x = [[0.2, 0.3], [1.5, 1.7], [0.8, 0.9]]
x = np.array([[0.2, 0.3], [1.5, 1.7], [0.8, 0.9]])
mean = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])

# 计算概率密度函数
pdf = multivariate_gaussian_pdf(x, mean, cov)
logpdf = np.log(pdf)
print(pdf)
print(logpdf)