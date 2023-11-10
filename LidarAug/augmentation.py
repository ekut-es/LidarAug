import torch
from LidarAug import transformations
# import LidarAug
pc = torch.rand([1, 10000, 3])
print(pc)
transformations.translate(pc, torch.tensor([1.0, 2.0, 3.0]))
print(pc)
