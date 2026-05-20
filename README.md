# Restart — Phase 0: RL cold-load pickup, single feeder

A working proof that a learned policy can restore an 8-segment 10 kV feeder
after a cold-soak blackout faster and more safely than fixed-dwell baselines.

This is the **Phase 0** scaffold for the Restart project: an OpenDSS twin, a
realistic ETP+CLPU load model, a Gymnasium environment, two baselines, and a
MaskablePPO trainer. The numbers below are from a 150k-step training run on a
laptop CPU.

## Result (50 held-out scenarios, ambient -5 to +5 °C, outage 4–12 h)

| policy             | trip rate | restore rate | mean steps to 99% | mean peak trunk A |
| ------------------ | --------- | ------------ | ----------------- | ----------------- |
| greedy             | 100%      | 0%           | n/a               | 257               |
| sequential dwell=4 | 24%       | 76%          | 31                | 234               |
| sequential dwell=8 | 0%        | 100%         | 55                | 226               |
| **PPO (masked)**   | **0%**    | **100%**     | **15**            | **224**           |

The trained policy finishes restoration **~4× faster** than the safe fixed-dwell
baseline while matching its zero-trip safety.

## Layout

```
restart-grid-rl/
├── feeders/
│   └── feeder8.dss          OpenDSS model of the 8-segment 10 kV feeder
├── restart/
│   ├── etp.py               ETP load population + CLPU multiplier
│   ├── feeder.py            OpenDSS wrapper (snapshot solve)
│   ├── env.py               Gymnasium environment
│   ├── baselines.py         Greedy and Sequential reference policies
│   └── eval.py              Run any policy across seeds, compute metrics
├── scripts/
│   ├── train.py             Train a MaskablePPO policy
│   ├── evaluate.py          Compare baselines vs trained policy on held-out seeds
│   └── demo.py              Trace one restoration episode and save a PNG
├── results/                 Saved policies and plots
└── requirements.txt
```

## Quickstart

```bash
# (already done) python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/train.py    --steps 150000 --out results/policy.zip
.venv/bin/python scripts/evaluate.py --policy results/policy.zip --n 50
.venv/bin/python scripts/demo.py     --policy ppo --model results/policy.zip --seed 10042
.venv/bin/python scripts/demo.py     --policy sequential --seed 10042
.venv/bin/python scripts/demo.py     --policy greedy --seed 10042
```

Training takes ~3 minutes on a laptop CPU.

## How the physics works

Each of the 8 segments hosts a population of houses with thermostatic heaters.
While energized, thermostats cycle around their setpoints with hysteresis.
During the blackout, heaters are off and indoor temperatures drift toward
ambient. When power returns, every house whose temperature has dropped below
setpoint demands heat simultaneously — this is the cold-load pickup spike.

On top of the slow thermostatic dynamics, each segment carries a short-timescale
CLPU multiplier representing motor starts, refrigeration compressors, and
lighting surge:

```
load_kw(t) = etp_load_kw * (1 + clpu_mag * exp(-t_since_energized / clpu_tau))
```

`clpu_mag` is sampled uniformly in [1.0, 2.0] and `clpu_tau` in [60, 180] seconds
per segment, so different segments behave differently and the agent must adapt.

The trunk feeder line is monitored in OpenDSS; if its current exceeds `trip_amps`
(default 240 A), the upstream breaker trips and the episode ends with a -30
reward.

## How the RL works

**State** (19 floats): per-segment energization flag, per-segment cold-minutes,
normalized ambient temperature, trunk load p.u., fraction restored.

**Action** (Discrete, 9): close switch on segment 1–8 or wait one step.

**Action masking**: already-energized segments are masked out so the agent
cannot waste an action closing what's already closed.

**Reward** per step:

```
+10                          per segment newly energized this step
-1                           per step (urgency cost)
-2 * (0.93 - v_min_pu) * 10  if min bus voltage below 0.93 p.u.
-30                          on upstream trunk trip (terminal)
+30                          when all 8 energized without trip (terminal)
```

The -1 per-step cost is what prevents the agent from settling at "close some,
wait forever" — that strategy now accumulates -240 over a full episode and
loses to fast completion.

**Algorithm**: MaskablePPO from `sb3-contrib` with `MaskableActorCriticPolicy`,
`learning_rate=3e-4`, `n_steps=512`, `ent_coef=0.05`, `gamma=0.99`.

## What to do next

Phase 0 is a single-feeder proof. The interesting work starts now:

1. **More topologies.** Currently one 8-segment radial. Add the IEEE 13-bus
   and IEEE 34-bus models; condition the policy on a `topology_hash` so it
   generalizes across networks.

2. **Heterogeneous segments.** Today every segment has the same archetype.
   Real feeders mix residential, light commercial, and industrial — each with
   different CLPU shape. Sample archetypes and let the agent learn which to
   prioritize.

3. **Voltage ramp and dwell as explicit actions.** Currently the action is
   only "close segment i". Phase 1 should let the agent choose voltage ramp
   rate and explicit dwell — that's the actual operator-facing control.

4. **Disturbance scenarios.** Inject a second-event during restoration
   (further damage, frequency excursion from ENTSO-E neighbor) and see if
   the policy recovers.

5. **Curriculum.** Start training on short outages where greedy works,
   gradually lengthen to current 4–12 h range, and beyond into multi-day
   winter blackouts.

6. **Hardware-in-the-loop.** Export the trained policy and run against a
   Typhoon HIL bench with the same feeder model in real time. The first
   sim-to-real check.

7. **SHAP explainer.** Attach feature attributions so every recommendation
   carries a one-line "because cold_minutes is high on S4 and trunk_headroom
   is tight."

8. **Operator console.** Wrap the policy in a web UI that shows the
   recommendation, the explanation, and the expected vs. actual trunk current
   after each action.

## Caveats

- The OpenDSS snapshot solve is not transient — fast electromagnetic
  transients (sub-cycle inrush, motor starting impedances) are abstracted
  into the CLPU multiplier. Phase-1 should use a quasi-dynamic or full
  EMT solve for realistic transformer inrush.

- The ETP + CLPU parameters were chosen so the demo is interesting on a
  laptop. Real residential-archetype calibration against Ukrainian SCADA
  is the Phase-1 data engineering project.

- 150k training steps gives a policy good enough for this proof but is a
  fraction of what's needed for a topology-distribution-robust policy.
  Plan on 5–20M steps per regional twin for Phase 2.
