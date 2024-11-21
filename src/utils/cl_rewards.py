# MIT License
#
#@title Copyright (c) 2024 CCAI Community Authors { display-mode: "form" }
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
from typing import Any

from citylearn.reward_function import RewardFunction

class CostNoBattPenalization(RewardFunction):

    def __init__(self, env_metadata: dict[str, Any]):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(
        self, observations: list[dict[str, int, float]]
    ) -> list[float]:
        
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):

            cost = o['net_electricity_consumption'] * o['electricity_pricing']
            battery_soc = o['electrical_storage_soc']
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            reward = penalty*abs(cost)

            reward_list.append(reward)

        return [sum(reward_list)] if self.central_agent else reward_list
    
class CostBadBattUsePenalization(RewardFunction):

    def __init__(self, env_metadata: dict[str, Any]):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

        self.p_1 = 0.4 # Penalty within the SoC
        self.p_2 = 1 - self.p_1 # Penalty outside the SoC 
        self.previous_observations = None

    import numpy as np

    def sigmoid(self, x, k=10):
        """Smooth sigmoid function to replace conditions and max."""
        return 1 / (1 + np.exp(-k * x))

    def penalization_soft(self, action, soc, p1, p2, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0):
        """
        Refined penalization function for battery control scenario using sigmoid with minimal penalty
        for valid discharging actions when battery is charged, and focus on penalizing
        invalid actions like overcharging or over-discharging.
        
        Parameters:
        - action: Action taken, where -1 means full discharge, 1 means full charge.
        - soc: State of charge, where 0 means empty and 1 means fully charged.
        - p1: Penalization factor for actions within the SoC limits.
        - p2: Penalization factor for actions beyond the SoC limits.
        - alpha1, alpha2, alpha3, alpha4: Scaling factors for penalization terms.
        
        Returns:
        - Total penalization value.
        """
        
        # Smooth factor for transition between penalization (can be tuned)
        k = 10
        
        # Penalization for overcharging only when SoC exceeds 1
        p2_charge_full = alpha1 * action * self.sigmoid(action * (soc + action - 1), k) * (action - (1 - soc))**2 if soc + action > 1 else 0

        # Penalization for exceeding charge limit (when soc is less than 1)
        p2_charge_exceed = alpha2 * action * self.sigmoid(action - (1 - soc), k) * (action - (1 - soc))**2 if action > (1 - soc) else 0

        # Penalization for over-discharging only when SoC is less than 0 (avoiding big penalties when discharging a charged battery)
        p2_discharge_empty = alpha3 * (-action) * self.sigmoid(-(soc + action), k) * (soc + action)**2 if soc + action < 0 else 0

        # Penalization for discharging more than available energy, but ensuring minimal penalty when valid (e.g., SoC = 1 and action = -1)
        p1_discharge_exceed = alpha4 * action * self.sigmoid(-action - soc, k) * (soc + action)**2 if action < -soc else 0
        
        # Combine penalizations with weights p1 and p2
        total_penalization = p1 * p1_discharge_exceed + p2 * (p2_charge_full + p2_charge_exceed + p2_discharge_empty)
        
        return abs(total_penalization)[0] if isinstance(total_penalization, np.ndarray) else abs(total_penalization)

    def calculate(
        self, observations: list[dict[str, int, float]]
    ) -> list[float]:
        
        r"""Returns reward for most recent action.

        This reward is simply the result of the operation:

        $-E_{t,i}^{\text{grid}} = -\max\{ E_{t,i}^{\text{load}} + E^{\text{batt}}_{t,i} - E^{\text{solar}}_{t,i}, 0\}$

        Which account only when there is net energy consumption (grid is used)

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        # self.previous_observations = [{} for _ in self.env_metadata['buildings']] if self.previous_observations is None else self.previous_observations

        reward_list = []

        for i, (o, m) in enumerate(zip(observations, self.env_metadata['buildings'])):

            # Store the relevant previous observations

            # self.previous_observations[i]['net_electricity_consumption'] = o['net_electricity_consumption']
            # self.previous_observations[i]['electrical_storage_electricity_consumption'] = o['electrical_storage_electricity_consumption']
            # self.previous_observations[i]['electrical_storage_soc'] = o['electrical_storage_soc']
            # self.previous_observations[i]['electricity_pricing'] = o['electricity_pricing']
            # self.previous_observations[i]['solar_generation'] = o['solar_generation']
            
            # Compute penalty for battery mismanagement

            penalty = self.penalization_soft(
                action=self.env_metadata['last_action'][i],
                soc=o['electrical_storage_soc'],
                p1=self.p_1,
                p2=self.p_2
            )

            cost = -(o['net_electricity_consumption'] * o['electricity_pricing'])
            reward = cost.clip(max=0)

            reward_list.append(reward - penalty)

        return [sum(reward_list)] if self.central_agent else reward_list
    
class CostIneffectiveActionPenalization(RewardFunction):

    def __init__(self, env_metadata: dict[str, Any]):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(
        self, observations: list[dict[str, int, float]]
    ) -> list[float]:
        
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for i, (o, m) in enumerate(zip(observations, self.env_metadata['buildings'])):

            penalization = (self.env_metadata['last_action'][i][0] - o['electrical_storage_electricity_consumption']) ** 2

            cost = -1 * o['net_electricity_consumption'] * o['electricity_pricing']

            reward_list.append(cost - penalization)

        return [sum(reward_list)] if self.central_agent else reward_list
    
class Cost(RewardFunction):

    def __init__(self, env_metadata: dict[str, Any]):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(
        self, observations: list[dict[str, int, float]]
    ) -> list[float]:
        
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):

            cost = o['net_electricity_consumption'] * o['electricity_pricing']

            reward_list.append(cost)

        return [sum(reward_list)] if self.central_agent else reward_list
    
class WeightedCostAndEmissions(RewardFunction):

    def __init__(self, env_metadata: dict[str, Any]):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        cost_weight: float:
            Weight for the cost component of the reward.
        emissions_weight: float:
            Weight for the emissions component of the reward.
        """

        super().__init__(env_metadata)
        self.emissions_weight = 0.1
        self.cost_weight = 1 - self.emissions_weight
        self.last_soc = None

    def calculate(
        self, observations: list[dict[str, int, float]]
    ) -> list[float]:
        
        r"""Returns reward for most recent action.

        The reward is a weighted combination of electricity cost and emissions.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for i, (o, m) in enumerate(zip(observations, self.env_metadata['buildings'])):
            
            prev_soc = self.last_soc[i]
            battery_soc = o['electrical_storage_soc']
            cost = max(o['net_electricity_consumption'], 0) * o['electricity_pricing']
            cost += min(o['net_electricity_consumption'], 0) * o['electricity_pricing'] * self.env_metadata['price_margin']
            emissions = max(o['net_electricity_consumption'], 0) * o['carbon_intensity']
            last_action = self.env_metadata['last_action']

            # Handle baseline agent

            if len(last_action[0]) == 0:
                last_action = 0.0
            else:
                # Handle central agent
                last_action = last_action[0][i] if self.central_agent else last_action[i][0]

            # Penalize the agent for not using the battery effectively
            
            # penalty = 0.0
            # # pen_factor = 1e2

            # # Penalize if charging or discharging the battery beyond battery's limits

            # if (last_action > 0 and last_action > (1 - prev_soc)) or (last_action < 0 and abs(last_action) > prev_soc):
            #     # penalty += pen_factor * (abs((1 - prev_soc) - last_action) if last_action > 0 else abs(prev_soc + last_action))
            #     penalty += 5
                
            penalty = 1.0
            pen_factor = 1000

            # Penalize if charging or discharging the battery beyond battery's limits

            if (last_action > 0 and last_action > (1 - prev_soc)) or (last_action < 0 and abs(last_action) > prev_soc):
                penalty += pen_factor * (abs((1 - prev_soc) - last_action) if last_action > 0 else abs(prev_soc + last_action))

            # Accumulate  penalization for not using 

            penalty = -(penalty + np.sign(cost) * battery_soc)

            # Compute reward

            reward = -(self.cost_weight * cost + self.emissions_weight * emissions)
            # reward = -(self.cost_weight * cost + self.emissions_weight * emissions)
            # reward = - max(o['net_electricity_consumption'], 0)

            reward_list.append(reward + penalty)
            # reward_list.append(reward - penalty)
            reward_list.append(abs(reward) * penalty)
            # if reward > 0:
            #     reward_list.append(reward / penalty)
            # else:
            #     reward_list.append(penalty * reward)

            # Update previous state of charge

            self.last_soc[i] = o['electrical_storage_soc']

        return [np.mean(reward_list)] if self.central_agent else reward_list
        # return [np.sum(reward_list)] if self.central_agent else reward_list