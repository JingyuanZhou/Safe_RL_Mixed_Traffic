# Safe RL for Mixed-Autonomy Traffic
Code for the paper "Enhancing System-Level Safety in Mixed-Autonomy Platoon via Safe Reinforcement Learning" (IEEE Transaction on Intelligent Vehicles)
[[PDF](https://ieeexplore.ieee.org/document/10462535)]

![](assets/overview.pdf)

**Abstract:**

> Connected and automated vehicles (CAVs) have recently gained prominence in traffic research due to advances in communication technology and autonomous driving. Various longitudinal control strategies for CAVs have been developed to enhance traffic efficiency, stability, and safety in mixed-autonomy scenarios. Deep reinforcement learning (DRL) is one promising strategy for mixed-autonomy platoon control, thanks to its capability of managing complex scenarios in real time after sufficient offline training. However, there are three research gaps for DRL-based mixed-autonomy platoon control: (i) the lack of theoretical collision-free guarantees, (ii) the widely adopted but impractical assumption of skilled and rational drivers who will not collide with preceding vehicles, and (iii) the strong assumption of a known human driver model. To address these research gaps, we propose a safe DRL-based controller that can provide a system-level safety guarantee for mixed-autonomy platoon control. First, we combine control barrier function (CBF)-based safety constraints and DRL via a quadratic programming (QP)-based differentiable neural network layer to provide theoretical safety guarantees. Second, we incorporate system-level safety constraints into our proposed method to account for the safety of both CAVs and the following HDVs to address the potential collisions due to irrational human driving behavior. Third, we devise a learning-based human driver behavior identification approach to estimate the unknown human car-following behavior in the real system. Simulation results demonstrate that our proposed method effectively ensures CAV safety and improves HDV safety in mixed platoon environments while simultaneously enhancing traffic capacity and string stability.

### Preparation

```
pip install -r requirements.txt
```

### Run

1. Train a new model:
```
python main.py
```
2. Model evaluation:
```
python plot_training_traj.py
python visualize.py
python plot_traj.py
python SafeRegion.py
```
### Citation

> J. Zhou, L. Yan and K. Yang, "Enhancing System-Level Safety in Mixed-Autonomy Platoon via Safe Reinforcement Learning," in *IEEE Transactions on Intelligent Vehicles*, doi: 10.1109/TIV.2024.3373512.
