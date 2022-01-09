import numpy as np
import torch

#centroids = np.load('centroids/office-home/centroids.npy')

x = torch.Tensor([[5, 5, 5, 5], [2, 2, 2, 2], [.1, .1, .1, .1], [3, 3, 3, 3], [4, 4, 4, 4]])
y = torch.Tensor([[0, 0, 0, 0], [5, 5, 5, 5]]) # loop

#dist = torch.norm(y - x, dim=1, p=None)
#knn = dist.topk(1, largest=False)
#print(knn)
dist = torch.cdist(x, y, p=2)
knn = dist.topk(1, largest=False, dim = 0)
print(knn.indices)
"""data = torch.randn(100, 10)
test = torch.randn(1, 10)

dist = torch.norm(data - test, dim=1, p=None)
knn = dist.topk(3, largest=False)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))"""