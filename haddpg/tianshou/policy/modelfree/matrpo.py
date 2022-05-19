import torch
import warnings
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Type, Optional, Union
from torch.distributions import kl_divergence
from tianshou.utils import RunningMeanStd
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy


class MATRPOPolicy(BasePolicy):
    """Implementation of Trust Region Policy Optimization. arXiv:1502.05477.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param int optim_critic_iters: Number of times to optimize critic network per
        update. Default to 5.
    :param int max_kl: max kl-divergence used to constrain each actor network update.
        Default to 0.01.
    :param float backtrack_coeff: Coefficient to be multiplied by step size when
        constraints are not met. Default to 0.8.
    :param int max_backtracks: Max number of backtracking times in linesearch. Default
        to 10.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        num_agents,
        dist_fn: Type[torch.distributions.Distribution],
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        advantage_normalization: bool = True,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        # super().__init__(actor, critic, optim, dist_fn, **kwargs)
        super().__init__(action_scaling=action_scaling,
                         action_bound_method=action_bound_method, **kwargs)
        self.num_agents=num_agents
        self.actor = actor
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
    
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._batch = max_batchsize

        del self._weight_vf, self._weight_ent, self._grad_norm

        self._norm_adv = advantage_normalization
        self._optim_critic_iters = optim_critic_iters
        self._step_size = actor_step_size
        # adjusts Hessian-vector product calculation for numerical stability
        self._damping = 0.1

        self._max_backtracks = max_backtracks
        self._delta = max_kl
        self._backtrack_coeff = backtrack_coeff
        self._optim_critic_iters: int
        

    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s.append(self.critic(b.obs))
                v_s_.append(self.critic(b.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch, buffer, indice, v_s_, v_s,
            gamma=self._gamma, gae_lambda=self._lambda)
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        batch = self._compute_returns(batch, buffer, indice)
        batch.act = to_torch_as(batch.act, batch.v_s)
        # batch = super().process_fn(batch, buffer, indice)
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(b).dist.log_prob(b.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self._norm_adv:
            batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        return batch
    
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits_collector_0=[]
        logits_collector_1=[]
        for i in range(self.num_agents):

            logit, h = self.actor[i](batch.obs, state=state)
            logits_collector_0.append(logit[0])
            logits_collector_1.append(logit[1])
        logits_0=torch.cat(logits_collector_0,-1)
        logits_1=torch.cat(logits_collector_1,-1)
        logits=(logits_0,logits_1)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        actor_losses, vf_losses, step_sizes, kls = [], [], [], []
        for step in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                
                for agent_id in range(self.num_agents):
                    dist = self(b).dist  # TODO could come from batch
                    ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                    ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                    actor_loss = -(ratio * b.adv).mean()
                    flat_grads = self._get_flat_grad(
                        actor_loss, self.actor[agent_id], retain_graph=True).detach()

                    # direction: calculate natural gradient
                    with torch.no_grad():
                        old_dist = self(b).dist

                    kl = kl_divergence(old_dist, dist).mean()
                    # calculate first order gradient of kl with respect to theta
                    flat_kl_grad = self._get_flat_grad(kl, self.actor[agent_id], create_graph=True)
                    search_direction = -self._conjugate_gradients(
                        flat_grads, flat_kl_grad, self.actor[agent_id], nsteps=10)

                    # stepsize: calculate max stepsize constrained by kl bound
                    step_size = torch.sqrt(2 * self._delta / (
                        search_direction * self._MVP(search_direction, flat_kl_grad, self.actor[agent_id])
                    ).sum(0, keepdim=True))

                    # stepsize: linesearch stepsize
                    with torch.no_grad():
                        flat_params = torch.cat([param.data.view(-1)
                                                for param in self.actor[agent_id].parameters()])
                        for i in range(self._max_backtracks):
                            new_flat_params = flat_params + step_size * search_direction
                            self._set_from_flat_params(self.actor[agent_id], new_flat_params)
                            # calculate kl and if in bound, loss actually down
                            new_dist = self(b).dist
                            new_dratio = (
                                new_dist.log_prob(b.act) - b.logp_old).exp().float()
                            new_dratio = new_dratio.reshape(
                                new_dratio.size(0), -1).transpose(0, 1)
                            new_actor_loss = -(new_dratio * b.adv).mean()
                            kl = kl_divergence(old_dist, new_dist).mean()

                            if kl < self._delta and new_actor_loss < actor_loss:
                                if i > 0:
                                    warnings.warn(f"Backtracking to step {i}.")
                                break
                            elif i < self._max_backtracks - 1:
                                step_size = step_size * self._backtrack_coeff
                            else:
                                self._set_from_flat_params(self.actor[agent_id], new_flat_params)
                                step_size = torch.tensor([0.0])
                                warnings.warn("Line search failed! It seems hyperparamters"
                                            " are poor and need to be changed.")

                # optimize citirc
                for _ in range(self._optim_critic_iters):
                    value = self.critic(b.obs).flatten()
                    vf_loss = F.mse_loss(b.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                step_sizes.append(step_size.item())
                kls.append(kl.item())

        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "step_size": step_sizes,
            "kl": kls,
        }

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor,model:nn.Module) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(
            kl_v,  model, retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _conjugate_gradients(
        self,
        b: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        model:nn.Module,
        nsteps: int = 10,
        residual_tol: float = 1e-10
    ) -> torch.Tensor:
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()
        # Note: should be 'r, p = b - MVP(x)', but for x=0, MVP(x)=0.
        # Change if doing warm start.
        rdotr = r.dot(r)
        for i in range(nsteps):
            z = self._MVP(p, flat_kl_grad, model)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def _get_flat_grad(
        self, y: torch.Tensor, model: nn.Module, **kwargs: Any
    ) -> torch.Tensor:
        grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _set_from_flat_params(
        self, model: nn.Module, flat_params: torch.Tensor
    ) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model