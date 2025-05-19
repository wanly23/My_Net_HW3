import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)

class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)

# class MyNet(nn.Module):
#     def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
#         super().__init__()
#         self.device = device
#         self.config = config
#         # 提取棋盘尺寸
#         self.board_size = observation_size[0]  # 假设是正方形棋盘
        
#         # 简单的卷积网络结构 - 只用少量层
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
        
#         # 计算展平后的特征数量
#         flatten_size = 32 * self.board_size * self.board_size
        
#         # 策略头 - 简化版
#         self.policy_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(flatten_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_space_size)
#         )
        
#         # 价值头 - 简化版
#         self.value_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(flatten_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
        
#         self.to(device)
    
#     def forward(self, s: torch.Tensor):
#         # 确保输入形状正确
#         if len(s.shape) == 2:  # 单个棋盘 [H, W]
#             s = s.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#         elif len(s.shape) == 3:  # 批量棋盘 [B, H, W]
#             s = s.unsqueeze(1)  # [B, 1, H, W]
            
#         # 确保数据类型为浮点
#         if not s.is_floating_point():
#             s = s.float()
            
#         # 确保在正确设备上
#         s = s.to(self.device)
        
#         # 前向传播
#         x = self.conv_layers(s)
        
#         # 策略和价值输出
#         pi = self.policy_head(x)
#         v = self.value_head(x)
        
#         return F.log_softmax(pi, dim=1), torch.tanh(v)