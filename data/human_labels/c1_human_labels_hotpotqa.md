# C1 - human labels for the judge bake-off

**Gate (execution plan Step 1.6): the judge must agree with human labels on
>=90% of trajectories.** Only inter-judge agreement (93.3%) has been computed
so far, which does not test correctness.

## How to fill this in

For each trajectory below, decide **which single module is most responsible
for the wrong final answer** - i.e. which module's output, had it been correct,
would most likely have produced the right answer.

Write your choice on the `LABEL:` line. Valid values for hotpotqa:

```
generate_query     the search query was bad, so the right evidence was never retrieved
generate_answer    the evidence was adequate, but the answer drawn from it was wrong
unclear            genuinely cannot tell (excluded from scoring, not counted against the judge)
```

**The judge's answers are deliberately not shown.** Seeing them first would anchor
your labels and make the agreement figure meaningless. Do not look at
`analysis/results/judge_bakeoff.json` until you have finished.

When done: `python analysis/c1_score_labels.py --task hotpotqa`

---

## Trajectory 0

**Question:** Which band formed first Cha Cha Cohen or Swervedriver?

**Gold answer:** `Cha Cha Cohen were a band formed in 1994`

**Pipeline's answer:** `Cannot determine from provided context`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 1

**Question:** Who wrote lyrics for both "Little Shop of Horrors" and the song "Be Our Guest?" 

**Gold answer:** `Howard Elliott Ashman`

**Pipeline's answer:** `Howard Ashman`  (EM/F1 = 0.400)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 2

**Question:** Who passed away first Max Ophüls and Shirley Clarke ?

**Gold answer:** `Max Ophüls`

**Pipeline's answer:** `Cannot be determined from the provided context.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 3

**Question:** What animated movie, starring Danny Devito, featured music written and produced by Kool Kojak?

**Gold answer:** `The Lorax`

**Pipeline's answer:** `The context provided does not contain information about an animated movie starring Danny DeVito with music by Kool Kojak.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 4

**Question:** Who is older Carrie Akre or Shifty Shellshock?

**Gold answer:** `Carrie Akre`

**Pipeline's answer:** `Cannot determine from provided context`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 5

**Question:** Which physicist known for his work on parity violation has a son who became an American chemist teaching at Cornell University?

**Gold answer:** `Tsung-Dao Lee`

**Pipeline's answer:** `Cannot be determined from the provided context.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 6

**Question:** Clifford Odets was seen as a successor to a Nobel Prize winner known as what?

**Gold answer:** `American playwright and Nobel laureate in Literature`

**Pipeline's answer:** `Eugene O'Neill`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 7

**Question:** What is the name of the character portrayed by Anna Camp in the film centered on the fictional Barden University and The Bellas?

**Gold answer:** `Aubrey Posen`

**Pipeline's answer:** `The character name is not provided in the given context.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 8

**Question:** Where was the sports complex at which Will Brodie made his debut in the thirty-five point loss against Melbourne located?

**Gold answer:** `Alice Springs, Northern Territory`

**Pipeline's answer:** `Not provided in context`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 9

**Question:** In what year was the aviator in whose honour the Quai d'Auteuil was renamed born?

**Gold answer:** `1872`

**Pipeline's answer:** `Not provided in context`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 10

**Question:** Are The Maine and Black both from Tempe, Arizona?

**Gold answer:** `no`

**Pipeline's answer:** `Cannot determine from provided context`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 11

**Question:** Which artist has worked with King Crimson and also released a solo album in 2011?

**Gold answer:** `Steven Wilson`

**Pipeline's answer:** `Cannot be determined from the provided context.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 12

**Question:** Who has won more literary awards, Annie Dillard or Gabriel García Márquez?

**Gold answer:** `Gabriel José de la Concordia García Márquez`

**Pipeline's answer:** `Cannot be determined from the provided context.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 13

**Question:** The maternal grandmother of Princess Marie Cécile of Prussia was born in which year ?

**Gold answer:** `1876`

**Pipeline's answer:** `Cannot be determined from the provided context.`  (EM/F1 = 0.000)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---

## Trajectory 14

**Question:** Electronic Attack Squadron 138 is a type of which American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet?

**Gold answer:** `Boeing EA-18G Growler`

**Pipeline's answer:** `EA-18G Growler`  (EM/F1 = 0.400)

**Execution trace:**

```
  (no trajectory recorded -- re-run optimizers/judge_bakeoff.py)
```

LABEL: 

---
