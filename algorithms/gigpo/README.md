# GiGPO: Group-in-Group Policy Optimization

> **A critic-free reinforcement-learning algorithm for assigning both journey-level and decision-level credit to long-horizon LLM agents.**

This document explains GiGPO from first principles using one running example throughout:

> **Training an LLM agent to complete a simulated flight-booking journey.**

The example is illustrative. The original GiGPO paper evaluates environments such as ALFWorld, WebShop, and search-augmented question answering—not a production airline-booking system.

---

## 1. Objective

A flight-booking agent may perform many actions before the final outcome is known:

1. Enter origin
2. Enter destination
3. Select departure date
4. Select return date
5. Select passenger count
6. Search for flights
7. Apply the nonstop filter
8. Select an outbound flight
9. Select a return flight
10. Choose a fare class
11. Enter passenger information
12. Add baggage
13. Select a seat
14. Review the itinerary
15. Verify that all user constraints are satisfied

A final reward tells us whether the complete itinerary was acceptable.

But it does not directly tell us:

- Which flight-selection action was good?
- Which action caused a budget violation?
- Which step introduced an unnecessary detour?
- Which earlier actions were correct even though the final journey failed?

This is the **credit-assignment problem**.

GiGPO addresses it by giving every action two signals:

1. A grade for the **complete journey**
2. A grade for the **specific decision made from the same repeated state**

Its central equation is:

$$
\boxed{ A\left(a_t^{(i)}\right) = A^E(\tau_i) + \omega A^S\left(a_t^{(i)}\right) }
$$

Read this as:

> **Final credit for one action = whole-journey credit + weighted local-decision credit.**

---

## 2. The running flight-booking task

Assume the training task is:

```text
Book a simulated round-trip flight:

Origin: Seattle (SEA)
Destination: New York (JFK)
Dates: October 12–16
Passengers: 1 adult
Constraints:
    nonstop only
    total itinerary price <= $500
    one checked bag
    aisle seat

Safety:
    use frozen simulated inventory
    do not charge a card
    stop after itinerary verification
```

The environment is a simulator or replayable sandbox. No real reservation or payment is made.

This matters because GiGPO needs several independent attempts at the same task and the same initial state.

---

## 3. One policy, multiple rollouts—not multiple cooperating agents

GiGPO does **not** require four different trained agents to cooperate on one booking.

There is one LLM policy with one set of weights:

$$
\pi_\theta
$$

During training, that same policy is sampled several times in separate copies of the environment:

```text
One frozen LLM policy
        |
        +-- Simulator session 1 --> trajectory tau_1
        +-- Simulator session 2 --> trajectory tau_2
        +-- Simulator session 3 --> trajectory tau_3
        +-- Simulator session 4 --> trajectory tau_4
```

The sessions:

- receive the same booking instruction;
- begin from the same search page;
- use the same frozen flight inventory;
- use the same policy;
- do not communicate;
- may take different actions because sampling is stochastic.

The paper denotes the number of rollouts by:

$$
N
$$

For this explanation:

$$
N=4
$$

The paper used larger groups in its experiments, including group size 8 for WebShop.

---

## 4. Why ordinary GRPO is too coarse for this task

Suppose four booking attempts produce these outcomes:

| Trajectory | Behavior | Final result |
|---|---|---|
| $\tau_1$ | Selects a valid nonstop flight and finishes directly | Success |
| $\tau_2$ | Selects an invalid option, returns to results, repairs, then succeeds | Success with detour |
| $\tau_3$ | Selects a one-stop flight even though nonstop was required | Failure |
| $\tau_4$ | Selects another valid nonstop flight and finishes directly | Success |

Trajectory-level GRPO can tell us:

```text
tau_1 was good
tau_2 was mostly good
tau_3 was bad
tau_4 was good
```

But it gives the same trajectory advantage to every action inside a trajectory.

In $\tau_3$, these earlier actions may have been correct:

```text
Enter SEA
Enter JFK
Choose the correct dates
Choose one passenger
Apply the nonstop filter
```

Yet all of them inherit the same negative trajectory-level signal as the bad flight selection.

GRPO knows:

> “This complete journey failed.”

It does not know:

> “The outbound-flight selection was the local mistake.”

GiGPO adds a second comparison at repeated states to recover finer credit.

---

## 5. GiGPO in one sentence

> **GiGPO first compares complete journeys, then searches those journeys for repeated states and compares the actions taken from those same states.**

The name **Group-in-Group** comes from these two nested groups:

```text
Outer group:
    complete flight-booking trajectories

Inner groups:
    actions taken from the same repeated booking state
```

---

## 6. Symbols and terminology

Every symbol used below is defined here.

| Symbol | Meaning in the flight-booking example |
|---|---|
| $x$ | The booking instruction |
| $N$ | Number of independent rollouts for the same task |
| $i$ | Rollout index: 1, 2, 3, or 4 |
| $t$ | Decision-step index within a rollout |
| $s_t^{(i)}$ | Booking state at step $t$ in rollout $i$ |
| $a_t^{(i)}$ | Action selected from that state |
| $r_t^{(i)}$ | Immediate reward or penalty after the action |
| $\tau_i$ | Complete trajectory for rollout $i$ |
| $R(\tau_i)$ | Total return of the complete trajectory |
| $R_t^{(i)}$ | Discounted future return beginning at step $t$ |
| $A^E(\tau_i)$ | Episode-level advantage for the entire trajectory |
| $A^S(a_t^{(i)})$ | Step-level advantage for one action |
| $\gamma$ | Discount factor for delayed rewards |
| $\omega$ | Weight assigned to the step-level advantage |
| $F_{\text{norm}}$ | Optional advantage-normalization factor |
| $\pi_\theta$ | Current trainable LLM policy |
| $\pi_{\text{old}}$ | Frozen policy that generated the current rollouts |
| $\pi_{\text{ref}}$ | Reference policy used to control drift |
| $\rho_{i,t}$ | Current action probability divided by old action probability |
| $\epsilon_{\text{clip}}$ | PPO clipping threshold |
| $\beta$ | Strength of the reference-policy KL penalty |

A **trajectory** is the complete sequence:

$$
\tau_i = \left\{ (s_1^{(i)},a_1^{(i)},r_1^{(i)}), \ldots, (s_T^{(i)},a_T^{(i)},r_T^{(i)}) \right\}
$$

---

## 7. Step 1: collect four complete trajectories

At the start of one training iteration:

$$
\pi_{\text{old}} \leftarrow \pi_\theta
$$

This means:

> Freeze the current model and use that frozen snapshot to generate the training data for this iteration.

Now run the policy independently in four simulator sessions.

### Rollout 1: direct success

```text
Searches the correct route and dates
Applies nonstop filter
Selects qualifying outbound flight A
Selects qualifying return flight
Chooses valid fare
Adds one checked bag
Selects aisle seat
Verifies itinerary
```

Total return:

$$
R(\tau_1)=10
$$

### Rollout 2: mistake, backtrack, repair

```text
Reaches the same outbound-results page
Selects an option that violates the task
Receives an invalid-choice penalty
Returns to the same results page
Selects qualifying outbound flight A
Completes the itinerary successfully
```

Total return:

$$
R(\tau_2)=9.9
$$

### Rollout 3: failure

```text
Reaches the same outbound-results page
Selects a one-stop flight
Continues without correcting it
Final verifier rejects the itinerary
```

Total return:

$$
R(\tau_3)=0
$$

### Rollout 4: direct success

```text
Reaches the same outbound-results page
Selects qualifying outbound flight B
Completes the itinerary successfully
```

Total return:

$$
R(\tau_4)=10
$$

---

## 8. Step 2: compute episode-level advantage Aᴱ

The episode-level signal asks:

> How good was this complete booking journey compared with the other journeys for the same task?

First compute the mean trajectory return:

$$
\overline R_E = \frac{10+9.9+0+10}{4} = 7.475
$$

GiGPO defines episode advantage as:

$$
\boxed{ A^E(\tau_i) = \frac{ R(\tau_i)-\overline R_E }{ F_{\text{norm}} } }
$$

For this worked example, use:

$$
F_{\text{norm}}=1
$$

Then:

$$
A^E(\tau_1)=10-7.475=+2.525
$$

$$
A^E(\tau_2)=9.9-7.475=+2.425
$$

$$
A^E(\tau_3)=0-7.475=-7.475
$$

$$
A^E(\tau_4)=10-7.475=+2.525
$$

| Trajectory | Complete return | Episode advantage |
|---|---:|---:|
| $\tau_1$ | 10.0 | +2.525 |
| $\tau_2$ | 9.9 | +2.425 |
| $\tau_3$ | 0.0 | -7.475 |
| $\tau_4$ | 10.0 | +2.525 |

This is the global, journey-level grade.

It still does not identify which action in $\tau_3$ caused the failure.

---

## 9. Step 3: find repeated states—the anchor states

After the trajectories finish, GiGPO scans the stored states.

Assume all four rollouts reached the same outbound-flight results page:

```text
Origin: SEA
Destination: JFK
Dates: October 12–16
Passengers: 1
Filter: nonstop only
Maximum total budget: $500
Frozen inventory version: V17
No flight selected yet
```

Call this repeated state:

$$
\tilde s_{\text{outbound-results}}
$$

The tilde indicates that this is an **anchor state**: a repeated state used to build a local comparison group.

The actions observed from this state were:

| Occurrence | Action |
|---|---|
| $\tau_1$ | Select qualifying nonstop flight A |
| $\tau_2$, first visit | Select a constraint-violating option |
| $\tau_2$, second visit | Select qualifying nonstop flight A |
| $\tau_3$ | Select a one-stop flight |
| $\tau_4$ | Select qualifying nonstop flight B |

Notice that four trajectories produced **five action occurrences** because rollout 2 returned to the same state.

This is allowed. GiGPO groups matching states across different trajectories and across different times inside one trajectory.

The step group is:

$$
\boxed{ G^S(\tilde s) = \left\{ \left(a_t^{(i)},R_t^{(i)}\right) \;\middle|\; s_t^{(i)}=\tilde s \right\} }
$$

Read this as:

> Gather every action and its future return whenever the policy encountered anchor state $\tilde s$.

No new branches are generated from the state. The grouping is performed retroactively over trajectories that were already collected.

---

## 10. Step 4: compute discounted future return Rₜ⁽ⁱ⁾

The outbound-flight selection may receive no immediate final-success reward.

The agent must still:

- choose the return flight;
- choose a fare;
- enter passenger data;
- select baggage;
- select a seat;
- verify the itinerary.

Therefore GiGPO uses the discounted future return:

$$
\boxed{ R_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t}r_k^{(i)} }
$$

Every term means:

- $t$: the action being evaluated;
- $k$: a future step;
- $T$: the last step in the trajectory;
- $r_k^{(i)}$: reward at future step $k$;
- $\gamma$: discount factor;
- $\gamma^{k-t}$: how much a later reward counts for the current action.

Use:

$$
\gamma=0.95
$$

A reward received soon counts more than the same reward received after a long detour.

For this illustration, suppose:

- a direct qualifying selection reaches success six steps later;
- the initial bad selection in rollout 2 reaches success only after a long backtrack;
- the one-stop selection in rollout 3 ends in failure.

The future returns are approximately:

| Action occurrence | Future return |
|---|---:|
| Direct qualifying choice in $\tau_1$ | $10(0.95)^6=7.35$ |
| Initial bad choice in $\tau_2$ | $-0.1+10(0.95)^{12}=5.30$ |
| Correct retry in $\tau_2$ | $10(0.95)^6=7.35$ |
| One-stop choice in $\tau_3$ | 0.00 |
| Direct qualifying choice in $\tau_4$ | $10(0.95)^6=7.35$ |

---

## 11. Step 5: compute step-level advantage Aˢ

First compute the mean future return for actions from this same state:

$$
\overline R_S = \frac{7.35+5.30+7.35+0+7.35}{5} \approx 5.47
$$

The step-level advantage is:

$$
\boxed{ A^S\left(a_t^{(i)}\right) = \frac{ R_t^{(i)}-\overline R_S }{ F_{\text{norm}} } }
$$

Again use:

$$
F_{\text{norm}}=1
$$

Then:

| Action | Step advantage |
|---|---:|
| Direct qualifying selection | $7.35-5.47=+1.88$ |
| Initial bad selection followed by repair | $5.30-5.47=-0.17$ |
| Correct retry | $7.35-5.47=+1.88$ |
| One-stop selection followed by failure | $0-5.47=-5.47$ |
| Other direct qualifying selection | $7.35-5.47=+1.88$ |

The local signal now says:

```text
Select a qualifying nonstop flight directly:
    better than average from this state

Take a poor action but eventually repair:
    slightly worse than average

Select a one-stop flight and fail:
    much worse than average
```

This information was unavailable to trajectory-level GRPO.

---

## 12. Step 6: combine global and local credit

GiGPO combines the two advantages:

$$
\boxed{ A\left(a_t^{(i)}\right) = A^E(\tau_i) + \omega A^S\left(a_t^{(i)}\right) }
$$

Use:

$$
\omega=1
$$

Then:

| Action occurrence | $A^E$ | $A^S$ | Combined advantage |
|---|---:|---:|---:|
| Direct qualifying choice in $\tau_1$ | +2.525 | +1.88 | **+4.41** |
| Initial bad choice in $\tau_2$ | +2.425 | -0.17 | **+2.26** |
| Correct retry in $\tau_2$ | +2.425 | +1.88 | **+4.30** |
| One-stop choice in $\tau_3$ | -7.475 | -5.47 | **-12.95** |
| Direct qualifying choice in $\tau_4$ | +2.525 | +1.88 | **+4.41** |

Interpretation:

- Direct correct actions receive the strongest reinforcement.
- The detour action is reinforced much less than the correct retry because its local credit is worse.
- The action that violates the task and leads to failure receives a strongly negative signal.

An important limitation is visible here:

> A poor action inside an ultimately successful trajectory can still have positive total advantage because the successful episode advantage may dominate the negative local correction.

GiGPO provides finer relative credit, not perfect causal blame.

---

## 13. What does ω do?

$\omega$ controls how much local credit matters.

### ω = 0

$$
A=A^E
$$

The local signal disappears. The method reduces to trajectory-level group optimization.

### ω = 1

$$
A=A^E+A^S
$$

Global and local credit are combined directly.

### Large ω

Local comparisons dominate.

This can help when state matching is reliable, but it can hurt when:

- two states were incorrectly treated as identical;
- future returns are noisy;
- the environment failed after a correct action;
- the same visible page hides different inventory or fare conditions.

The paper reports that an intermediate value worked best in one WebShop sensitivity study, while values in a broader middle range remained reasonably competitive.

---

## 14. Step 7: update the policy with PPO-style clipping

GiGPO changes the advantage estimator, but it retains a PPO-style controlled update.

For one sampled action:

$$
\boxed{ \rho_{i,t} = \frac{ \pi_\theta\left(a_t^{(i)}\mid s_t^{(i)},x\right) }{ \pi_{\text{old}}\left(a_t^{(i)}\mid s_t^{(i)},x\right) } }
$$

This ratio asks:

> How much more or less likely is the sampled action under the current model than under the model that generated the rollout?

Example:

```text
Old probability of selecting qualifying flight A: 0.30
Current probability after updates:               0.39
```

Then:

$$
\rho=\frac{0.39}{0.30}=1.30
$$

The action became 30% more likely.

The clipped objective uses:

$$
\boxed{ \min\left( \rho_{i,t}A_{i,t}, \operatorname{clip} \left(\rho_{i,t},1-\epsilon_{\text{clip}},1+\epsilon_{\text{clip}}\right) A_{i,t} \right) }
$$

If:

$$
\epsilon_{\text{clip}}=0.2
$$

then the useful ratio range is approximately:

$$
[0.8,1.2]
$$

The model is encouraged to move in the correct direction, but it receives no extra benefit from changing too aggressively in one update.

A KL penalty also discourages excessive drift from a reference model:

$$
-\beta D_{\mathrm{KL}}\left(\pi_\theta\Vert\pi_{\text{ref}}\right)
$$

---

## 15. Complete GiGPO algorithm in plain language

```text
Initialize:

    pi_theta
        Current trainable LLM agent

    pi_ref
        Reference policy used to control drift

    task distribution p(X)
        Simulated flight-booking tasks

    N = 4
        Four independent rollout sessions per task

    gamma = 0.95
        Discount factor for delayed reward

    omega = 1
        Weight of local step advantage

    epsilon_clip = 0.2
        PPO clipping threshold

    beta
        KL-penalty strength

    Safe flight-booking simulator
        Frozen inventory and prices
        No real reservation
        No payment
        Stop after itinerary verification


For every training iteration:

    1. Freeze the current rollout policy:
           pi_old <- pi_theta

    2. Sample one booking task x.

    3. Initialize N isolated but identical simulator sessions:
           same task
           same inventory snapshot
           same initial page
           same constraints

    4. Run pi_old independently in all N sessions.

       At every step, record:
           task x
           current state s_t
           generated action a_t
           immediate reward r_t
           next state s_(t+1)
           old action-token log probabilities

    5. Store the N complete trajectories:
           tau_1, ..., tau_N

    6. Compute each trajectory return:
           R(tau_i) = sum of rewards in trajectory i

    7. Compute episode advantages:
           A_E(tau_i)
             = trajectory return
               - average return of trajectories for the same task

       Optionally divide by a normalization factor.

    8. Canonicalize the states and scan all trajectories
       for repeated states.

    9. For every repeated anchor state s_tilde:
           collect every action taken from that state

   10. For every collected action:
           compute discounted future return R_t^(i)

   11. Compute step advantages:
           A_S
             = action's future return
               - average future return of actions from the same state

       Optionally divide by a normalization factor.

   12. For every action, combine:
           A = A_E + omega * A_S

   13. Compute current-policy / old-policy probability ratios.

   14. Update pi_theta using:
           PPO-style clipping
           combined GiGPO advantage
           reference-policy KL penalty

       Positive A:
           increase action probability

       Negative A:
           decrease action probability

   15. Discard the old rollout batch.

   16. Generate fresh trajectories with the updated policy.

   17. Repeat.
```

---

## 16. Compact implementation-oriented pseudocode

```python
for iteration in range(num_iterations):
    old_policy = freeze_copy(policy)
    task = task_distribution.sample()

    trajectories = []

    for rollout_id in range(group_size):
        env = simulator.reset(
            task=task,
            inventory_snapshot=task.inventory_snapshot,
        )

        trajectory = []

        while not env.terminal:
            state = canonicalize_state(env.observe())
            action, old_logprobs = old_policy.sample_action(state, task)
            next_state, reward, metadata = env.step(action)

            trajectory.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=canonicalize_state(next_state),
                    old_logprobs=old_logprobs,
                    metadata=metadata,
                )
            )

        trajectories.append(trajectory)

    episode_returns = [sum_rewards(tau) for tau in trajectories]
    episode_advantages = relative_advantages(episode_returns)

    anchor_groups = group_transitions_by_state(trajectories)

    step_advantages = {}
    for state_key, occurrences in anchor_groups.items():
        if len(occurrences) == 1:
            step_advantages[occurrences[0].id] = 0.0
            continue

        future_returns = [
            discounted_return(occ.trajectory, occ.step, gamma)
            for occ in occurrences
        ]

        local_advantages = relative_advantages(future_returns)

        for occ, local_adv in zip(occurrences, local_advantages):
            step_advantages[occ.id] = local_adv

    losses = []

    for trajectory_id, trajectory in enumerate(trajectories):
        for transition in trajectory:
            total_advantage = (
                episode_advantages[trajectory_id]
                + omega * step_advantages.get(transition.id, 0.0)
            )

            current_logprobs = policy.logprobs(
                transition.state,
                transition.action,
                task,
            )

            ratio = exp(current_logprobs - transition.old_logprobs)

            clipped_ratio = clip(
                ratio,
                1.0 - clip_epsilon,
                1.0 + clip_epsilon,
            )

            policy_term = minimum(
                ratio * total_advantage,
                clipped_ratio * total_advantage,
            )

            kl_term = beta * kl_divergence(policy, reference_policy)
            losses.append(-(policy_term - kl_term))

    optimizer.zero_grad()
    mean(losses).backward()
    optimizer.step()
```

This is conceptual pseudocode, not a drop-in training implementation. In an LLM implementation, action probabilities are normally represented through token-level log probabilities and appropriate action-token masks.

---

## 17. How GiGPO differs from PPO and GRPO

| Question | PPO | GRPO | GiGPO |
|---|---|---|---|
| How is expected reward estimated? | Learned critic | Group statistics over complete outputs or trajectories | Episode-group statistics plus repeated-state action-group statistics |
| Critic required? | Yes | No | No |
| Main advantage granularity | State/token level | Complete response or trajectory | Complete trajectory plus repeated-state action level |
| Multiple rollouts per task? | Not inherently required | Yes | Yes |
| Repeated-state grouping? | No | No | Yes |
| PPO-style clipping? | Yes | Usually yes in GRPO implementations | Yes |
| Main strength | Potentially fine state-specific credit | Simple critic-free optimization | Finer agent credit without a critic or extra per-state rollouts |
| Main weakness | Critic cost and instability | Coarse credit in long trajectories | Depends on reliable state matching and repeated states |

A concise relationship is:

$$
\boxed{ \text{GiGPO} = \text{GRPO-style episode credit} + \text{same-state step credit} }
$$

---

## 18. State matching is the central implementation challenge

Two flight-results pages should be grouped only when they represent the same decision problem.

A useful canonical state may include:

```text
origin
selected destination
outbound date
return date
passenger count
cabin
active filters
sort order
inventory snapshot ID
visible flight options
selected outbound segment
selected return segment
fare family
baggage state
seat-map state
validation errors
modal or popup state
```

It should exclude irrelevant noise such as:

```text
session ID
timestamp
animation frame
random DOM node ID
tracking parameters
```

### False merge

These states look similar but are not equivalent:

```text
Same search page, but prices have changed
Same flight list, but one fare sold out
Same seat map, but different seat availability
Same results, but nonstop filter is active in only one state
```

Grouping them together corrupts $A^S$.

### False split

These states represent the same decision but differ only in irrelevant data:

```text
Different session identifier
Different timestamp
Different random element IDs
```

Treating them as different creates singleton groups, causing:

$$
A^S=0
$$

and GiGPO falls back to trajectory-level credit for those states.

For real airline systems, inventory and prices are dynamic. A frozen simulator or replayable inventory snapshot is therefore essential for controlled GiGPO training.

---

## 19. Reward design for the flight example

A simple simulated reward could be:

| Event | Reward |
|---|---:|
| Final itinerary satisfies all constraints | +10.0 |
| Final itinerary violates a required constraint | 0.0 |
| Invalid or unavailable action | -0.1 |
| Exceeds price limit | Negative penalty |
| Chooses connecting itinerary when nonstop was required | Negative penalty |
| Correctly applies a required filter | Small verified reward |
| Unnecessary backtrack | Small penalty |
| Attempts to make a real purchase | Large negative safety penalty |

Rewards should be based on verified environment state, not on the model merely claiming that an action succeeded.

For example:

```text
Bad reward:
    Agent emitted "apply nonstop filter"

Good reward:
    Environment confirms nonstop filter is active
```

Similarly:

```text
Bad reward:
    Agent said it chose an aisle seat

Good reward:
    Seat-map state confirms an aisle seat is selected
```

---

## 20. Important failure cases

### 20.1 The state appears only once

If an anchor group contains one occurrence:

$$
R_t-\operatorname{mean}(R_t)=0
$$

Therefore:

$$
A^S=0
$$

The action receives only episode-level credit.

### 20.2 Every rollout makes the same mistake

Suppose all four rollouts select an invalid itinerary and receive zero:

$$
[0,0,0,0]
$$

There is no relative signal:

$$
A^E=0
$$

If all actions from the repeated state also have zero future return:

$$
A^S=0
$$

GiGPO cannot invent a successful behavior when the group contains no useful variation.

Possible remedies include:

- supervised fine-tuning first;
- better demonstrations;
- more exploration;
- a curriculum of easier tasks;
- denser verified intermediate rewards.

### 20.3 The environment fails after a correct action

Suppose the policy chooses a valid flight, but the simulator or browser tool fails to register the selection.

If the trajectory is scored as an ordinary policy failure, GiGPO may suppress the correct action.

An implementation should distinguish:

```text
Policy decision:
    Was the semantic choice correct?

Tool execution:
    Did the environment execute it?

State verification:
    Did the selected flight actually persist?
```

Environment failures should be censored or handled separately rather than mislabeled as reasoning failures.

### 20.4 Dynamic inventory breaks state equivalence

If fare availability changes across sessions, the sessions are no longer identical. Same-state comparison becomes invalid.

Use frozen or versioned inventory during training.

### 20.5 GiGPO is not root-cause analysis

GiGPO learns relative action utility from returns.

It does not prove:

```text
The first causal failure was exactly step 8.
```

Deterministic validators and trajectory diagnostics remain useful.

---

## 21. What a training implementation must record

For every transition:

```text
task ID
rollout ID
step ID
canonical state key
raw observation reference
action text or structured action
action-token mask
old action-token log probabilities
immediate reward
next state
terminal flag
discounted return
episode return
episode advantage
step-group ID
step advantage
combined advantage
execution status
state-verification status
```

For every training batch, monitor:

```text
success rate
mean episode return
fraction of zero-variance episode groups
fraction of singleton step groups
fraction of repeated states
anchor-group size distribution
mean and variance of A_E
mean and variance of A_S
PPO clipping fraction
KL from reference policy
action entropy
average trajectory length
constraint-violation rate
environment-failure rate
```

---

## 22. Minimal evaluation plan

Compare at least:

1. Base instruction model
2. Supervised fine-tuned model
3. GRPO
4. GiGPO without step advantage: $\omega=0$
5. Full GiGPO
6. PPO, when critic cost is acceptable

Keep fixed:

```text
initial model
task split
reward function
simulator inventory
sampling temperature
maximum steps
optimizer
reference policy
rollout-token budget
```

Report:

```text
constraint-satisfying itinerary rate
success at fixed action budget
average number of steps
unnecessary-backtrack rate
invalid-action rate
price-limit violation rate
nonstop-constraint violation rate
seat and baggage accuracy
GPU memory
rollout tokens
wall-clock training time
```

The most important ablation is:

$$
\omega=0 \quad\text{versus}\quad \omega>0
$$

This isolates whether same-state step credit adds value over trajectory-level group credit.

---

## 23. When GiGPO is a good fit

GiGPO is attractive when:

- tasks require multiple sequential decisions;
- rewards are sparse or delayed;
- multiple safe rollouts can be generated for one task;
- repeated states naturally occur;
- states can be matched reliably;
- final and intermediate outcomes can be verified;
- training a separate critic is undesirable.

Examples include:

- simulated flight booking;
- web navigation;
- form completion;
- search agents;
- software agents;
- embodied environments;
- interactive games.

---

## 24. When GiGPO is a poor fit

GiGPO is less suitable when:

- only one live execution is possible;
- actions have irreversible real-world consequences;
- repeated states rarely occur;
- state equivalence cannot be established;
- all rollouts receive the same reward;
- environment failures dominate policy failures;
- generating several complete trajectories is too expensive.

A production airline-booking system should not create several real reservations merely to form a training group. Training must use simulation, replay, or intercepted transactions.

---

## 25. Final takeaway

PPO asks a critic to estimate expected future reward for every state.

GRPO removes the critic and compares complete trajectories:

```text
Which complete journeys were better?
```

GiGPO keeps that global comparison and adds a local one:

```text
When several journeys reached the same booking state,
which action from that state produced the better future?
```

The central equation is:

$$
\boxed{ A\left(a_t^{(i)}\right) = A^E(\tau_i) + \omega A^S\left(a_t^{(i)}\right) }
$$

For the flight-booking example:

```text
A_E:
    Was the complete itinerary-building journey good?

A_S:
    From this exact flight-results state, was this flight choice
    better than the other choices actually tried?

omega:
    How strongly should that local comparison affect training?
```

The shortest useful mental model is:

> **GRPO tells the model which booking journeys were better. GiGPO also tells it which choices were better when multiple journeys reached the same decision point.**

---

## Reference

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. **Group-in-Group Policy Optimization for LLM Agent Training.** NeurIPS 2025, arXiv:2505.10978.

The equations and core algorithm in this README follow the paper. The flight-booking scenario and numerical adaptation are explanatory examples created for this guide.

