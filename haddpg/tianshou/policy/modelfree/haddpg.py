import torch
import warnings
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Batch, ReplayBuffer


class HADDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action. Default to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        Default to False.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: Optional[torch.nn.Module],
        actor_optim: list,#Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        num_agents,
        action_shape,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",\
        action_idx: list = [],
        **kwargs: Any,
    ) -> None:
        super().__init__(action_scaling=action_scaling,
                         action_bound_method=action_bound_method, **kwargs)
        self.num_agents=num_agents
        self.action_dim= int(action_shape)
        self.action_idx = action_idx
        assert action_bound_method != "tanh", "tanh mapping is not supported" \
            "in policies where action is used as input of critic , because" \
            "raw action in range (-inf, inf) will cause instability in training"
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            for i in range(self.num_agents):
                self.actor_old[i].eval()
            #self.actor_optim: torch.optim.Optimizer = actor_optim
            self.actor_optim = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self._tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rew_norm = reward_normalization
        self._n_step = estimation_step

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        for i in range(self.num_agents):
            self.actor[i].train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for i in range(self.num_agents):
            for o, n in zip(self.actor_old[i].parameters(), self.actor[i].parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            
    def sync_weight_critic(self) -> None:
        for o, n in zip(self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            
    def sync_weight_actor(self, i) -> None:
        for o, n in zip(self.actor_old[i].parameters(), self.actor[i].parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            
    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        target_q = self.critic_old(
            batch.obs_next,
            self(batch, model='actor_old', input='obs_next').act)
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions_collector=[]

        for i in range(self.num_agents):
            action, h = model[i](obs, state=state, info=batch.info)
            actions_collector.append(action)
        actions=torch.cat(actions_collector,-1)
        return Batch(act=actions, state=h)

    @staticmethod
    def _mse_optimizer(
        batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic
        td, critic_loss = self._mse_optimizer(
            batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actions_collector=[]
        actor_loss = 0
        device = self.critic.device
        for i in range(self.num_agents):
            actions = torch.tensor(batch.act.copy()).to(device).detach()
            obs = batch.obs.copy()
            action, h = self.actor[i](obs, info=batch.info)
            actions[:,i*self.action_dim:(i+1)*self.action_dim]=action[:]
            actor_loss += -self.critic(batch.obs, actions).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
    
    def ddpg_update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], train_times: int, **kwargs: Any
    ) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        # collect samples
        batch_all = []
        indice_all = []
        for i in range(train_times):
            if buffer is None:
                return {}
            batch, indice = buffer.sample(sample_size)
            batch = self.process_fn(batch, buffer, indice)
            #self.post_process_fn(batch, buffer, indice)
            batch_all.append(batch)
            indice_all.append(indice)
            
        # update critic function
        for i in range(train_times):
            batch = batch_all[i]
            indice = indice_all[i]
            self.updating = True
            td, critic_loss = self._mse_optimizer(
                    batch, self.critic, self.critic_optim)
            batch.weight = td
            self.post_process_fn(batch, buffer, indice)
            #self.sync_weight_critic()
            
        device = self.critic.device
        actions_all = [torch.tensor(batch.act.copy()).to(device).detach() for batch in batch_all]
        obs_all = [batch.obs.copy() for batch in batch_all]
        
        # print(actions_all[0][:10])
        # update agents sequentially
        for agent_id in torch.randperm(self.num_agents):
            # print(agent_id)
            for i in range(train_times):
                obs = obs_all[i]
                actions = actions_all[i].clone().detach()

                action, h = self.actor[agent_id](obs, info=batch_all[i].info)
                actions[:,self.action_idx[agent_id]:self.action_idx[agent_id+1]]=action[:]
                actor_loss = -self.critic(obs, actions).mean()

                self.actor_optim[agent_id].zero_grad()
                actor_loss.backward()
                self.actor_optim[agent_id].step()
            
            # update original batch with new actions

            for i in range(train_times):
                obs = obs_all[i]
                actions = actions_all[i]
                with torch.no_grad():
                    action, h = self.actor[agent_id](obs, info=batch_all[i].info)
                actions[:,self.action_idx[agent_id]:self.action_idx[agent_id+1]]=action[:]         
                
        self.sync_weight()
        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }
        self.updating = False
        return result

