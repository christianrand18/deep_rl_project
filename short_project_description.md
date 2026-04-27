### Project Overview: Hierarchical Reinforcement Learning and Brand Momentum in Competitive AMoD Systems

This project extends the competitive multi-operator Autonomous Mobility-on-Demand (AMoD) framework by introducing a **two-level Hierarchical Reinforcement Learning (HRL) architecture** and a **passenger loyalty proxy**. To manage computational complexity while capturing long-term strategic competition, we will simulate sequential daily peak-demand periods (1 hour per day). For example, an 8-hour total simulation time will represent the peak hour across 8 consecutive days.

To achieve this, we will introduce a daily **meta-policy** that guides high-level strategy and modify the passenger discrete choice model to account for **brand momentum**, rewarding operators who consistently capture market share day over day.

---

### Core Components of the Extension

#### 1. Two-Level MDP and Meta-Policy Architecture
We will frame the problem as a two-level Markov Decision Process (MDP) operating across multi-day episodes. A higher-level "meta-policy" network will operate on a daily frequency, updating once after each 1-hour peak simulation. The lower-level policy will continue to operate on 3-minute time steps within that hour.
* **Inputs:** The meta-policy will ingest aggregated daily statistics from the lower-level policy, including service rates, average prices (own and competitor), rebalancing volume/costs, and regional profits.
* **Outputs/Action Space:** The meta-policy will guide the low-level actor for the following day through one of two proposed methods:
    * *Reward Shaping:* Outputting a scaling parameter (e.g., $\alpha$) that modifies the lower-level agent's reward function based on the difference between chosen prices and reference prices.
    * *Direct Constraints:* Outputting zone-wise price scalars to explicitly constrain or guide the low-level pricing strategy.
* **Training:** We will utilize PPO or an Actor-Critic architecture, initially testing the meta-policy on a single operator before expanding to a fully competitive setup. 

#### 2. Brand Momentum and Passenger Loyalty
The original discrete choice model allocates demand purely based on immediate trip price, travel time, and passenger wage. Because the simulation lacks persistent passenger IDs to track individual loyalty, we will introduce "brand momentum" as a system-level proxy that carries over between days.
* **Mechanism:** An operator's utility function will be upweighted dynamically based on their recent market dominance. 
* **Implementation:** We will use a function of the proportion of demand served over the last $n$-days (e.g., via exponential smoothing). For example, capturing **80%** of the market in recent daily peak hours will provide a utility bonus in the upcoming days' discrete choice calculations, simulating word-of-mouth popularity or platform habituation.

#### 3. Evaluation and Post-Hoc Analysis
By integrating mathematical psychology concepts and hierarchical RL, the post-hoc analysis will focus on how long-term strategic planning and brand momentum alter the competitive equilibrium over a multi-day horizon. We will investigate whether operators choose to run short-term losses (e.g., heavy undercutting on early days) to build brand momentum for long-term profitability on subsequent days, compared to the baseline where competition relies solely on short-term spatial positioning and immediate price competition.