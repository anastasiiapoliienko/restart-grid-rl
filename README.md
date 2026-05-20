# Restart — Phase 0: RL cold-load pickup, single feeder

A working proof that a learned policy can restore an 8-segment 10 kV feeder
after a cold-soak blackout dramatically faster than fixed-dwell baselines —
with a real safety tradeoff that motivates Phase 1.

This is the **Phase 0** scaffold for the Restart project: an OpenDSS twin, a
realistic ETP + CLPU load model, a Gymnasium environment, fixed-dwell baselines,
and a MaskablePPO trainer.

## Headline result

Evaluated on **1000 held-out scenarios** (ambient -5 to +5 °C, outage 4–12 h)
with a **3-seed PPO ensemble** (each trained 750k steps):

| policy           | trip rate         | restore rate      | mean steps to 99%  | mean peak A    |
| ---------------- | ----------------- | ----------------- | ------------------ | -------------- |
| greedy           | 100%              | 0%                | n/a                | 256            |
| sequential d=4   | 17.5%             | 82.5%             | 31                 | 232            |
| sequential d=8   | 0.3%              | 99.7%             | 55                 | 224            |
| sequential d=12  | **0.0%**          | **100%**          | 79                 | 222            |
| **PPO (3-seed)** | **9.2 % ± 1.6 %** | **90.8 % ± 1.6 %**| **11.5 ± 0.0**     | **232 ± 0.3**  |

### Reading this honestly

**The good:** the trained policy finishes restoration **~5–7× faster** than the
safest fixed-dwell baseline (11.5 steps vs 79). Across three independently
trained seeds the speed is identical to one decimal place — the policy
converged to a very stable, fast restoration strategy.

**The honest:** the same policy trips on ~9% of held-out scenarios. The earlier
50- and 100-seed evaluations showed 0% trips, but with 1000 scenarios we sample
the long-tail of cold-ambient + long-outage combinations where the policy
mis-judges the load. **This is exactly the speed-vs-safety frontier you would
expect from a pure-RL agent with no safety filter.**

The Phase-1 next step is therefore obvious: wrap the policy in a **safety
shield** — a thin rules layer that vetoes any closure expected to push trunk
current above a learned-risk threshold. PPO chooses the next *candidate*
action; the shield checks it against a fast load forecast and forces a wait
if the predicted trunk current crosses the limit. This combines PPO's speed
with sequential-dwell-12's perfect safety.

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
│   ├── train.py             Train one MaskablePPO policy
│   ├── evaluate.py          Compare baselines vs a single trained policy
│   ├── evaluate_multi.py    Compare baselines vs an N-policy ensemble
│   ├── plot_comparison.py   Violin/bar comparison plot
│   └── demo.py              Trace one restoration episode and save a PNG
├── results/                 Saved policies and plots
└── requirements.txt
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Train 3 independent seeds (3 × ~17 min on a laptop CPU)
for s in 1 2 3; do
    .venv/bin/python scripts/train.py --steps 750000 --seed $s \
        --out results/policy_s${s}.zip
done

# Headline evaluation on 1000 held-out scenarios
.venv/bin/python scripts/evaluate_multi.py \
    --policies results/policy_s1.zip results/policy_s2.zip results/policy_s3.zip \
    --n 1000

# Comparison plot (results/comparison.png)
.venv/bin/python scripts/plot_comparison.py \
    --policies results/policy_s1.zip results/policy_s2.zip results/policy_s3.zip \
    --n 500

# Single-scenario trace for a presentation
.venv/bin/python scripts/demo.py --policy ppo --model results/policy_s1.zip \
    --seed 10042 --out results/demo_ppo.png
```

## Physics

Each of the 8 segments hosts a population of houses with thermostatic heaters.
While energized, thermostats cycle around their setpoints with hysteresis.
During the blackout, heaters are off and indoor temperatures drift toward
ambient. When power returns, every house whose temperature has dropped below
setpoint demands heat simultaneously — the slow component of cold-load pickup.

On top of the slow thermostatic dynamics, each segment carries a short-timescale
CLPU multiplier representing motor starts, refrigeration compressors, and
lighting surge:

```
load_kw(t) = etp_load_kw * (1 + clpu_mag * exp(-t_since_energized / clpu_tau))
```

`clpu_mag` is sampled uniformly in [1.0, 2.0] and `clpu_tau` in [60, 180] seconds
**per segment**, so segments behave differently and the agent must adapt rather
than memorize a fixed sequence.

The trunk feeder line is monitored in OpenDSS; if its current exceeds
`trip_amps` (default 240 A), the upstream breaker trips and the episode ends
with a -30 reward.

## RL setup

**State** (19 floats): per-segment energization flag, per-segment cold-minutes,
normalized ambient temperature, trunk load p.u., fraction restored.

**Action** (Discrete, 9): close switch on segment 1–8, or wait one step.

**Action masking**: already-energized segments are masked so the agent can't
waste an action closing a switch that's already closed.

**Reward** per step:

```
+10                            per segment newly energized this step
-1                             per step (urgency cost)
-2 * (0.93 - v_min_pu) * 10    if min bus voltage below 0.93 p.u.
-30                            on trunk trip (terminal)
+30                            when all 8 energized without trip (terminal)
```

The -1 per-step cost is what prevents the agent from settling at
"close some, wait forever" — that strategy accumulates -240 over an episode
and loses to fast completion.

**Algorithm**: MaskablePPO (sb3-contrib) with `MaskableActorCriticPolicy`,
`learning_rate=3e-4`, `n_steps=512`, `ent_coef=0.05`, `gamma=0.99`,
750k steps per seed.

## What's next, in priority order

The 9% trip rate is the most important finding. Everything below points at
either reducing it directly or at making the proof more credible.

**1. Safety shield (1 week, highest leverage).** A two-line forecast model
predicts trunk current 30 s ahead under the agent's proposed action; if it
exceeds the trip threshold, the action is replaced with `wait`. This should
take PPO from 9% trips to essentially 0% with minimal speed loss — the
strongest argument we can make to a control-room engineer.

**2. Curriculum and domain randomization (3–4 days).** Today every training
episode samples uniformly across outage durations and ambient temperatures.
Bias training toward harder scenarios (deeper cold, longer outage) and
randomize the CLPU parameters per *episode* not per *segment* — the policy
has been over-fitting to the segment-level distribution.

**3. Heterogeneous segments (2–3 days).** Currently every segment is the same
residential archetype. Add commercial and small-industrial archetypes with
different CLPU shapes; the policy should learn to close light segments earlier
and heavy ones later. This is the "it learns priority" story.

**4. Second topology (1 week).** Drop the IEEE 13-bus model into `feeders/`
and condition the policy on a `topology_hash`. The "generalizes across
networks" claim is what every reviewer will ask about next.

**5. Voltage ramp and dwell as explicit actions (2–3 days).** Change the
action from `Discrete(N+1)` to `MultiDiscrete([N+1, 4, 4])` — choose segment,
ramp rate, dwell. This is the actual control an operator gives, and it makes
the agent's output directly translatable to a substation command.

**6. Operator console UI (1 week).** Single-file HTML page that shows the
agent's recommendation, the trunk-current forecast for the next 60 s if
accepted, and a one-line "why" (which features drove the choice). This is
what you put in front of an actual Ukrenergo or DSO contact.

**7. SHAP explainer (3–4 days).** Hook gradient-based feature attribution
into the policy network. Operator trust depends on this; no autonomous mode
is ever shipped without it.

**8. Hardware-in-the-loop (1 month, gated by partner access).** Once partners
are engaged, validate on Typhoon HIL or RTDS. This is the sim-to-real gate
before any shadow-mode talk.

## Caveats

- The OpenDSS snapshot solve is not transient — fast electromagnetic
  transients (sub-cycle inrush, motor starting impedances) are abstracted
  into the CLPU multiplier. Phase 1 should use a quasi-dynamic or full
  EMT solve.
- The ETP + CLPU parameters were chosen so the demo is interesting on a
  laptop. Real residential-archetype calibration against Ukrainian SCADA
  is the Phase-1 data engineering project.
- 750k training steps × 3 seeds is enough to *lock the speed* but not
  enough to close the 9% trip gap. The safety shield (item 1 above) is
  the right next attack, not "train for 10× longer".
