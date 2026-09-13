# Reinforcement Learning Algorithms

A first-principles collection of reinforcement-learning algorithms used in LLM post-training and multi-step agent training.

Each algorithm document is designed to connect four things:

1. **The problem the algorithm solves**
2. **The intuition behind the algorithm**
3. **The mathematical equations**
4. **A complete worked example and implementation-oriented pseudocode**

## Algorithms

| Algorithm | Main idea | Document |
|---|---|---|
| **GiGPO** | Combine whole-trajectory credit with same-state, step-level credit for long-horizon agents—without a critic | [Read the GiGPO guide](algorithms/gigpo/README.md) |

## Planned additions

- PPO — Proximal Policy Optimization
- GRPO — Group Relative Policy Optimization
- RLOO — REINFORCE Leave-One-Out
- DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization

## Repository philosophy

The goal is not to copy paper equations without explanation. Every document should answer:

- What was wrong with the previous method?
- Why was the new algorithm needed?
- What does every symbol mean?
- How does one training iteration work?
- What happens numerically in a concrete example?
- When does the algorithm fail?
- What must an implementation record and verify?

