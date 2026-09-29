# Understanding the C. elegans Oxygen Models

A guide to what Model 0 and Model 1 are doing, what the source paper claims, and how to read the results.

Sep 29, 2026 · @Someone

## The biological question

Put enough *C. elegans* worms on a food plate and they stop being evenly spread — they gather into clumps, ridges and swirling aggregates. Demir, Yaman, Basaran and Kocabas reported this in eLife in 2020 and asked what organises it.

The obvious answer would be that worms attract each other, or that each one steers toward its neighbours. Neither is what they found. The organiser is **oxygen**.

Three observations set up the whole project:

- Worms consume oxygen, so a crowd depletes the oxygen around itself.
- Worms change how fast they crawl depending on the oxygen they sit in. They prefer roughly 5-12% O2 and move faster outside that range, which in ordinary air (21% O2) means moving fast.
- The strain matters. Gray et al. showed in 2004 that *npr-1* mutants aggregate strongly while the standard N2 strain mostly does not, and that the difference tracks how sharply each strain responds to oxygen.

Gray et al. also ran the experiment that matters most here: they dropped the ambient oxygen from 21% to 7% and the aggregates came apart; they restored 21% and the aggregates re-formed. **Nothing about the animals changed — only the air above them.**

That is the question both models exist to answer: can consuming oxygen and changing speed in response to it, with no attraction and no steering, be enough to produce all of this?

## Why slow animals pile up

This is the one idea the whole project rests on. **Things that move slowly spend more time where they are slow, so they accumulate there** — without anyone deciding to gather.

The everyday version: traffic. Cars are densest where they move slowest. No driver is attracted to the jam; they just take longer to get through it. In steady state the density of anything wandering at a position-dependent speed goes like 1 / V, where V is the local speed.

Now close the loop. Suppose speed depends on oxygen, and oxygen is depleted by the animals themselves:

1. A patch happens to have slightly more worms than average — just random fluctuation.
2. More worms consume more oxygen, so the local oxygen drops.
3. Lower oxygen means slower worms (on the part of the response curve where speed rises with oxygen).
4. Slower worms leave more slowly, so the patch gets denser still.
5. Return to step 2.

That is a **positive feedback loop**, and it is why a uniform sheet of worms is unstable. A small ripple grows into a pattern.

### The one condition that has to hold

The loop only closes if step 3 works — if *less oxygen means slower*. On a sigmoid speed curve that is true only on the rising part. Sit the animals too high on the curve (plenty of oxygen, speed already near its ceiling) or too low (speed already at its floor) and a change in oxygen barely changes speed. The loop opens and nothing happens.

The models express this as a single number, **beta**:

```latex
\beta = \frac{V(O)\,V'(O)}{2 D_r}
```

V is the speed at the local oxygen level, V' is the slope of the speed curve there, and D\_r is how fast an animal forgets its heading. If V' is zero — flat part of the curve, or an animal whose speed ignores oxygen — then beta is zero and no pattern can form at any density. That is exactly what the `CONSTANT_SPEED` control strain tests.

**Nothing in this is steering.** No agent measures an oxygen gradient or moves toward anything. Each one reads the oxygen where it stands, sets a speed, and picks a random direction. The apparent "attraction" is a statistical consequence of speed varying in space.

## Model 0: isolating the feedback

Model 0 asks one question: **can that loop alone produce aggregation?** It is deliberately stripped down, so that if clumps appear there is nothing else they could be coming from.

It has four moving parts:

| Part | What it does |
| --- | --- |
| Agents | Points that move at speed V in a direction they slowly forget. No bodies, no collisions — they pass straight through each other. |
| Speed law | A sigmoid: V rises smoothly from `v_min` to `v_max` as local oxygen rises, set by a midpoint and a width. |
| Oxygen field | A grid that diffuses, is replenished from the air above at rate f, and is eaten by agents. |
| Consumption | Each agent removes oxygen over a small Gaussian footprint, saturating so it can never drive oxygen below zero. |

An agent's entire decision procedure each step is: read the oxygen here, set speed from it, take a step, jiggle the heading. That is all.

### What is deliberately absent

No attraction, no alignment, no collisions, no gradient sensing, no bacteria, no food depletion, no body mechanics. The heading marks in the plots are directions, not worm bodies. **If patterns form, the oxygen-speed loop is the only candidate.**

### What it showed

They do form. At Model 0's default settings the density heterogeneity H rises from about 0.03 to about 0.86 over 300 time units, while two controls stay flat at about 0.026:

- turn off the speed response (constant speed) — nothing happens;
- turn off consumption — nothing happens.

Both controls are necessary: the loop needs *both* directions of the coupling, agents changing oxygen and oxygen changing agents. Break either and it opens.

Model 0 is a released teaching package. Model 1 copies it in byte-for-byte and never modifies it, so it serves as a regression baseline — Model 1's engine is checked to reproduce Model 0's trajectories to about 1 part in 10^14.

## Model 1: freeze the animals, control the world

Model 1 asks a sharper question: **given a strain whose parameters you cannot touch, what can you do to it with the environment alone?**

That framing is the whole design. A real experimenter cannot edit a worm's oxygen response — they can only change the plate and the air. So Model 1 takes every agent parameter away from you and hands you the environment instead.

| Frozen into the strain | Yours, changeable mid-run |
| --- | --- |
| Speed floor and ceiling, response midpoint and width | Ambient oxygen: a level, a spatial map, or a time schedule |
| How fast a heading is forgotten (D\_r) | Replenishment rate f: a scalar or a map — a cover glass is a patch of low f |
| Consumption rate and saturation constant | Oxygen diffusivity |
| Consumption footprint width | The number of animals, added or removed at any moment |

### The strains

- **NPR1\_LIKE** — steep oxygen response, patterns readily. Stands in for the aggregating *npr-1* mutant.
- **N2\_LIKE** — weaker response, needs higher density before anything happens. Stands in for solitary N2.
- **CONSTANT\_SPEED** — the control. Speed ignores oxygen entirely, so beta is zero and it must never pattern.
- **MODEL0\_DEFAULT** — Model 0's exact settings, kept as the regression anchor.
- **HYPOXIC\_BRANCH** — optional, and it turns out to matter a lot. See the surprise below.

These are teaching presets. They are named after real strains because they reproduce a qualitative contrast, **not because any number in them was measured in a worm**.

### The rule that keeps it honest

The environment never writes to the oxygen field directly. It acts only through the replenishment term:

```latex
\frac{\partial O}{\partial t}\bigg|_{\text{env}} = f(x,y)\,\bigl(O_{\text{am}}(x,y,t) - O\bigr)
```

So a "low-oxygen spot" is a place where the gas being exchanged against is poor — not a place where oxygen is deleted. The animals' own consumption still competes with it. You can see this working: an aggregate sitting inside an imposed spot of level 0.15 pulls the actual local oxygen down to 0.02. If the environment simply overwrote the field, the feedback that makes patterns would be switched off, and the model would be answering a different question.

## Two halves that never talk to each other

Model 1 answers the question twice, by two routes that share no code, and then puts one on top of the other. That overlay is the actual result, and it only means anything because neither half saw the other.

### Half one: predict, without simulating

`stability.py` takes the agent rule to its continuum limit and asks whether a small ripple grows. It never runs a simulation. You give it a strain, an ambient level and a density; it returns whether the uniform state is unstable, and at what wavelength.

The machinery is standard linear stability analysis. Write density and oxygen as a flat state plus a small wave, keep only terms linear in the wave, and you get a 2x2 matrix per wavenumber k whose largest eigenvalue is the growth rate:

```latex
\lambda(k) = \text{largest eigenvalue of} \begin{pmatrix} -D_W k^2 & -\beta W k^2 \\ -a\,g(k) & -(D_O k^2 + f + b) \end{pmatrix}
```

If the largest growth rate over all k is positive, patterns should appear. Two things are worth noticing:

- The destabilising entry carries **beta**. If beta is zero, nothing is unstable at any density — the control strain, proved rather than observed.
- lambda(0) = 0 exactly, because the number of animals is conserved. A uniform shift of the density is not a growing mode.

### Half two: measure, without predicting

`patterns.py` looks at a density field and describes it. It never consults the prediction. Four numbers do the work:

1. **Heterogeneity H** — how lumpy the field is, compared against the lumpiness a purely random scatter of the same number of animals would show. This matters: a noisy field is lumpy too, so H is judged against its own noise floor, not a fixed constant.
2. **Structure factor** — the spatial Fourier spectrum, giving a characteristic length to compare against the predicted wavelength.
3. **Area fraction** — what share of the plate the dense phase occupies. This is what separates dots from stripes from holes.
4. **Topology on a torus** — how many separate dense blobs and dilute voids there are. The domain wraps around, so a stripe going all the way round is one object, not two; getting that wrong would corrupt every label.

Out comes one of four labels: `uniform`, `dots`, `stripes`, `holes`.

### Why the split matters

Linear theory predicts **whether**, not **what**. Dots, stripes and holes are all the same instability; which one you get is set by how much area the dense phase takes up, and that is measured, not derived. So the prediction cannot be quietly tuned to match the observation, and the agreement between them is a real test rather than a restatement.

## How to read a phase diagram

A phase diagram is a map of outcomes over two knobs. Here the horizontal axis is **ambient oxygen** (1.0 = ordinary air, 21% O2) and the vertical axis is **density** in animals per unit area. Each square is one condition, run three times with different random seeds.

- **Square colour** = the label most seeds agreed on.
- **Square opacity** = how many seeds agreed. A washed-out square is one the seeds disputed.
- **Shaded band** = where linear theory predicts instability, drawn independently of every simulation on the plot.

The figures live at `assets/phase_diagram_<strain>.png`.

### Why the unstable region is a diagonal stripe

This surprises people. You might expect "more density, more patterns". Instead the band runs corner to corner, because **ambient oxygen and density substitute for each other**.

What actually has to be true is that the *equilibrium oxygen* sits on the rising part of the speed curve. Two ways to get it there:

- lower the ambient oxygen, or
- raise the density, so the animals consume more of it.

Either one lowers the local oxygen. So a low-ambient, low-density plate and a high-ambient, high-density plate can land in the same place. That is why the band is diagonal, and it is exactly the trade-off the paper describes.

### Why the band has a ceiling as well as a floor

Less obvious, and it explains several results. Push the density high enough and the equilibrium oxygen falls *past* the rising part of the curve onto the lower plateau, where speed is already at its floor. There, changing oxygen barely changes speed — beta collapses again and patterns stop.

So each strain has a **window** of density, not a threshold. Too little density and the loop never engages; too much and it disengages from the other side.

### What "adjacent to the boundary" means

The headline agreement numbers exclude squares next to the predicted boundary. That is not massaging the result — it is stated up front and for a concrete reason. At the boundary the predicted growth rate goes to zero, so a condition that is formally unstable may have had time to grow nothing at all within the run; and the predicted wavelength diverges, so whatever does grow may not fit inside the box. Both effects are properties of the measurement, not of the prediction.

It does cost something: because the band is diagonal, a one-square margin removes 40 of 64 squares. The honest presentation is to give both numbers, which the results section does.

## What the runs actually showed

512 simulations across three strains, each on a 176 x 176 plate for 1200 time units, with 9,000 to 37,000 animals depending on density.

| Strain | Runs | Labels seen | Agreement away from the boundary | All 64 squares | Seed agreement |
| --- | --: | --- | --: | --: | --: |
| NPR1\_LIKE | 192 | uniform, dots, stripes, holes | 24/24 = 100% | 84.4% | 0.99 |
| N2\_LIKE | 192 | uniform, dots, stripes | 19/19 = 100% | 93.8% | 0.98 |
| CONSTANT\_SPEED | 128 | uniform only | 64/64 = 100% | 100% | 1.00 |

The target was 90%. Reading the diagram in one sentence: **every patterned square but one falls inside the predicted band**, and the squares that disagree are uniform ones lying inside the band near its ceiling, where the growth rate is going to zero.

### The control worked

`CONSTANT_SPEED` was uniform in all 128 runs. Its lumpiest run reached 1.05 times the random-scatter noise floor — statistically indistinguishable from a random scatter. Since these animals still eat oxygen and still sit in a structured oxygen field, and only the *feedback onto speed* is cut, this isolates beta specifically rather than "oxygen dynamics" in general.

### The strain contrast held up

The presets were tuned using linear theory alone, before any simulation. The sweep then confirmed the ordering independently:

| Ambient | NPR1 onset (observed) | NPR1 (predicted) | N2 onset (observed) | N2 (predicted) |
| --: | --: | --: | --: | --: |
| 0.70 | 0.30 | 0.21 | 0.43 | 0.41 |
| 0.80 | 0.30 | 0.27 | 0.56 | 0.48 |
| 0.90 | 0.30 | 0.34 | 0.56 | 0.57 |
| 1.00 | 0.43 | 0.42 | 0.69 | 0.66 |

Observed onsets land within roughly one grid step of predicted ones, and N2 needs 1.4 to 1.9 times the density at every ambient level — the *npr-1* versus N2 contrast, reproduced.

### Dots, then stripes, then holes

At ambient 1.0, sweeping density upward:

| Density | 0.30 | 0.43 | 0.56 | 0.69 | 0.81 | 0.94 | 1.07 | 1.20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label | uniform | dots | dots | dots | stripes | stripes | **holes** | uniform |

As the dense phase takes up more area it stops being isolated blobs, becomes connected ridges, and finally becomes a connected sheet with isolated voids punched through it. The `uniform` at the top is the ceiling effect — past the band.

Two honest caveats. **Holes are rare**: one square out of 64, three runs out of 192, because the band closes from above before the dense phase gets much past 70% of the area. And the hole label depends on how the dense/dilute threshold is chosen — with a simpler mean-density threshold no square in the entire sweep reads as holes, even though the topology (one connected dense region riddled with 24-40 isolated voids) is not in doubt. Both readings are recorded in the output files so the choice can be audited.

## The surprise: hypoxia freezes instead of dispersing

The most interesting result came from an experiment that looked, at first, like a bug.

The test was Gray et al.'s protocol: grow a pattern in air, drop the ambient oxygen to 7%, then restore it. The prediction is clear — at 7% the equilibrium oxygen falls off the rising part of the curve, beta collapses, so the aggregate should come apart. The first run reported that the pattern **never dissolved**.

It was not a bug. Lengthening the hypoxic window and adding a second strain gave this:

| Strain | H before | H after 2500 units of hypoxia | Time to disperse |
| --- | --: | --: | --: |
| NPR1\_LIKE (monotone speed curve) | 1.33 | 0.53 | never |
| HYPOXIC\_BRANCH (extra low-oxygen branch) | 0.66 | 0.021 | about 100 |

### Why

Dropping the oxygen does remove the driving force. But with a speed curve that only ever *rises* with oxygen, it also puts **every** animal at its speed floor. Speed falls from about 1.5 to about 0.068, and how fast a pattern can smear out goes like speed squared:

```latex
D_W = \frac{V(O)^2}{2 D_r}
```

So the ability to rearrange drops by a factor of about 480. Erasing a pattern is a spreading process, and it now needs roughly 1900 time units where before it needed a handful. **Hypoxia does not disperse the aggregate; it paralyses it.** The measured decay from 1.33 to 0.53 over 2500 units is exactly the partial relaxation that timescale implies.

### What fixes it

The paper reports that worms speed up *again* at very low oxygen — the speed curve is not monotone, it has a second high-speed branch down at the hypoxic end. `HYPOXIC_BRANCH` adds it. That strain removes the driving force **and keeps the animals mobile** (296 times the mobility at 2% O2), and its pattern collapses to the noise floor in about 100 time units and re-forms in about 90.

So reproducing the dispersal Gray et al. observed **requires** the non-monotone branch. A monotone speed law cannot do it in principle, not merely in practice. That is a concrete limitation of Model 0's speed law, inherited by Model 1's default presets.

### The same cause explains two other results

Once you see it, it accounts for a lot:

- **Hysteresis.** Ramping ambient oxygen up and back down at fixed population traces a large loop — a factor of 24 in lumpiness at the same ambient level. That looks like bistability, but it is not. The up-ramp lags because growth near the boundary is slow; the down-ramp persists because erasure at low oxygen is slow. Both arms are the clock, not the thermodynamics, and a slower ramp would narrow the loop. The write-up says so rather than claiming a genuine phase transition.
- **Spatial gradients.** Under an oxygen gradient the density should follow 1/V. It does in the oxygen-rich half (correlation +0.87) and does the opposite in the poor half (−0.63) — because equilibrating the poor half would take roughly 187,000 time units against a run of 1,500.

The phase diagram itself is unaffected, because it starts uniform in the oxygen-rich regime and only ever has to *grow* a pattern, never erase one.

## What this does and does not establish

Worth being blunt, because the figures look more authoritative than the claims underneath them.

**Nothing here is calibrated to worms.** Units are arbitrary. The presets were tuned with linear theory to give a clear contrast, not fitted to data. The strain names mark a qualitative resemblance, nothing more.

**Aggregation is not phase coexistence.** A high lumpiness score, a peak in the spectrum and a `dots` label together show the density field is structured at a measurable length. They do not show that two thermodynamic phases coexist. No coexistence densities were measured and no free energy was constructed. The labels are descriptions of shape.

**Some thresholds are judgement calls.** The dots/holes area-fraction cutoffs and the lumpiness gate were set once and held fixed. The dense/dilute separator was chosen *after* seeing output — that is disclosed in the validation notes with the numbers both ways, and the label rule itself was not touched. Labels near a threshold are not stable, and `stripes` also covers disordered labyrinths.

**The agreement figure is a consistency check, not a validation of the theory.** It measures how often a finite, noisy, time-limited simulation lands on the same side of a line as an infinite, deterministic, instantaneous prediction — on one grid, one box size, one run length, three seeds.

**Run lengths are finite.** Every "this did not pattern" means "did not pattern within the time given". Section 8 shows how much that matters.

**Out of scope entirely**, and therefore untested: bacteria, food depletion, the banding and swarming behaviours from later in the paper, body mechanics, collisions, alignment, reversals, gradient steering, and any comparison against measured worm data.

One piece is unfinished: the inverse-control milestone — steering an aggregate along a path with a moving low-oxygen spot — was running and was stopped part way. The search had got the tracking error down from 24.7 to 9.5 and was still improving.

## Where things live

If you want to poke at it, the notebook `01_environment_controlled_patterns.ipynb` is the guided tour — it runs in a few seconds and has a live demo where you drag the ambient oxygen and watch patterns form and dissolve.

| File | What it holds |
| --- | --- |
| `celegans/model0/` | Model 0, copied byte-for-byte and never edited |
| `celegans/model1/strains.py` | The frozen strain presets and the speed curves |
| `celegans/model1/environment.py` | Ambient levels, maps, schedules, spots, gradients, cover glass |
| `celegans/model1/engine.py` | The simulation, with live environment and live population |
| `celegans/model1/stability.py` | The prediction half — never runs a simulation |
| `celegans/model1/patterns.py` | The measurement half — never consults the prediction |
| `celegans/model1/sweep.py` | Parallel sweeps, resumable |
| `celegans/model1/control.py` | Inverse control and the CMA-ES search |
| `assets/` | Precomputed results the notebook loads: sweep CSVs and figures |
| `VALIDATION.md` | Every check run for this release, with numbers and limits |

The useful first look is `assets/phase_diagram_NPR1_LIKE.png` beside `assets/step_response.png` — the first is the main result, the second is the surprise.

To re-run things:

```bash
python -m unittest discover -s tests -v          # 57 checks, ~4 seconds
python scripts/phase_diagram.py --strain NPR1_LIKE   # ~30 min on 9 cores
python scripts/experiments.py --all              # the step, hysteresis, gradient runs
```

A warning worth heeding: the sweeps and experiments are hours of compute, not minutes. That is why the notebook loads precomputed results instead of generating them.
