from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
       # select the best action based on PUCB when expanding the tree
        exploration_denominator = node.child_N_visit + 1e-8 # 防止分母等于0
        total_sum = node.child_N_visit.sum()
        weights = np.zeros(node.n_action, dtype=np.float32)
        
        # PUCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(总访问次数) / (1 + N(s,a))
        exploitation = np.divide(node.child_V_total, node.child_N_visit + 1e-5)

        exploration = self.config.C * node.child_priors * np.sqrt(total_sum) / (exploration_denominator + 1)
        # 接口错误

        weights = exploitation + exploration
        weights = np.where(node.action_mask, weights, -INF)
        return np.argmax(weights)

    def backup(self, node:MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        current = node
        while current.parent is not None:
            action = current.action
            current.parent.child_N_visit[action] += 1
            current.parent.child_V_total[action] += value
            value = -value  # 切换玩家视角
            current = current.parent 
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        node = self.root

        while True:
            if node.done:
                return node
                
            legal_actions = np.where(node.action_mask)[0]
            unexpanded_actions = [a for a in legal_actions if not node.has_child(a)]
            
            # 优先扩展未访问的合法动作
            if unexpanded_actions:
                action = np.random.choice(unexpanded_actions)
                new_node = node.add_child(action)
                
                # 对于新节点，使用模型预测先验概率
                if not new_node.done:
                    obs = new_node.env.observation
                    child_prior, _ = self.model.predict(new_node.env.compute_canonical_form_obs(obs, new_node.env.current_player))
                    new_node.set_prior(child_prior)
                
                return new_node
            
            # 所有动作已扩展时使用PUCT选择
            else:
                action = self.puct_action_select(node)
                node = node.get_child(action)

        # return self.root
    
    def get_policy(self, node:MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        if node is None:
            node = self.root
            
        masked_visits = np.where(node.action_mask, node.child_N_visit, 0)
        total = masked_visits.sum()
        
        if total == 0:
            # 如果所有动作都未访问，返回均匀分布
            return np.ones(len(node.child_N_visit)) / len(node.child_N_visit)
        else:
            # 否则根据访问次数计算策略
            return masked_visits / total

    def search(self):
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                value = leaf.reward
            else:
                obs = leaf.env.observation
                _, value = self.model.predict(leaf.env.compute_canonical_form_obs(obs, leaf.env.current_player))
                value = value[0]
                value = -value
                # NOTE: you should compute the policy and value 
                #       using the value&policy model!
            self.backup(leaf, value)
            
        return self.get_policy(self.root)