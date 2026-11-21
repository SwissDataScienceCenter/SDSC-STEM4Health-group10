# Bioreactor Process Model — Research & Design Document

## 1. Dataset Analysis

### Project 1 — DTB (Deep Tank Bioreactor)
**Likely organism:** CHO cells (Chinese Hamster Ovary) — mammalian, but we can adapt the metabolic principles for microbial systems.

| Variable | Description | Range |
|---|---|---|
| `vcd` | Viable Cell Density | 0.86 – 39.38 ×10⁶ cells/mL |
| `tcd` | Total Cell Density | 0.86 – 40.11 ×10⁶ cells/mL |
| `substrate` | Main nutrient (glucose/glutamine) | 0.31 – 5.0 g/L |
| `byproduct` | Metabolic waste (lactate/ammonia) | 0.23 – 3.03 g/L |
| `elapsed_time` | Batch time | 0 – 13 days |
| `volume` | Reactor volume | 10.3 – 17.2 L |
| `feed_substrate` | Fed-batch nutrient feed | Sparse (132 rows) |
| `feed_b/c/d` | Additional feeds | Sparse |
| `to_predict` | Holdout flag | 156 rows for testing |

**Key observations:**
- Fed-batch process with sparse feed events
- 61 batches, 2858 observations (uneven time sampling)
- Cell growth → substrate depletion → byproduct accumulation
- Viability (vcd/tcd) declines toward end (dead cell accumulation)

### Project 2 — BRPMM (BioReactor Process Monitoring & Maintenance)
**Industrial process monitoring** — likely microbial fermentation (E. coli / yeast).

| Variable | Description |
|---|---|
| `flow_rate_1` | Main feed flow rate (g/min) |
| `density` | Broth density |
| `flow_rate_2` | Secondary flow |
| `pressure_1–4` | Pressure sensors (mbar) |
| `temperature_1–4` | Temperature sensors (°C) |

415K+ observations from 161 batches (121 accepted, 40 rejected).
This is an industrial monitoring/quality control process.

---

## 2. Microbial Metabolism — Key Processes & Equations

### 2.1 Primary Metabolism (Central Carbon)

#### Glycolysis (EMP Pathway)
Glucose → 2 Pyruvate + 2 ATP + 2 NADH
- **KEGG:** map00010
- **Rate-limiting enzyme:** Phosphofructokinase (PFK)

#### TCA Cycle (Krebs Cycle)
Acetyl-CoA → CO₂ + NADH + FADH₂ + GTP
- **KEGG:** map00020
- **Key:** Generates reducing equivalents for respiration

#### Pentose Phosphate Pathway
Glucose-6-P → Ribose-5-P + NADPH
- **KEGG:** map00030
- **Role:** Biosynthesis precursors + NADPH

### 2.2 Fermentation Pathways (Microbial Byproducts)

| Organism | Type | Substrates | Products |
|---|---|---|---|
| *E. coli* | Mixed acid | Glucose | Acetate, lactate, ethanol, succinate, CO₂, H₂ |
| *S. cerevisiae* | Ethanolic | Glucose | Ethanol, CO₂, glycerol |
| *Lactobacillus* | Lactic acid | Glucose | Lactate (homofermentative) |
| *C. acetobutylicum* | ABE | Glucose | Acetone, butanol, ethanol |
| *Z. mobilis* | Ethanolic | Glucose | Ethanol, CO₂ |
| *B. subtilis* | 2,3-BD | Glucose | 2,3-butanediol, acetoin, lactate |
| *C. glutamicum* | Glutamate | Glucose | L-glutamate, lactate, succinate |
| *P. pastoris* | Respiration | Methanol/glycerol | Recombinant protein, CO₂ |
| *A. niger* | Citric acid | Glucose | Citrate |
| *S. coelicolor* | Secondary | Glucose | Antibiotics, polyketides |

### 2.3 Respiration & Energy Metabolism

**Aerobic respiration:**
Glucose + 6O₂ → 6CO₂ + 6H₂O + 38 ATP (theoretical max)

**Electron Transport Chain:**
NADH → Complex I → Q → Complex III → Cyt c → Complex IV → O₂
FADH₂ → Complex II → Q → ...

**P/O ratios (ATP per oxygen atom):**
- Bacteria: 0.5–1.5
- Yeast: 1.0–1.5

---

## 3. Key Kinetic Models & Equations

### 3.1 Growth Kinetics

| Model | Equation | Parameters | Notes |
|---|---|---|---|
| **Monod** (1949) | μ = μ_max × S/(K_s + S) | μ_max, K_s | Most common, simple |
| **Moser** | μ = μ_max × S^n/(K_s + S^n) | μ_max, K_s, n | Variable cooperativity |
| **Contois** | μ = μ_max × S/(K_s·X + S) | μ_max, K_s | High-density cultures |
| **Andrews** (substrate inhibition) | μ = μ_max × S/(K_s + S + S²/K_i) | μ_max, K_s, K_i | Substrate inhibition |
| **Aiba** (product inhibition) | μ = μ_max × S/(K_s + S) × e^(-K_p·P) | μ_max, K_s, K_p | Ethanol/lactate inhibition |
| **Haldane** | μ = μ_max / (1 + K_s/S + S/K_i) | μ_max, K_s, K_i | Toxic substrates |
| **Double Monod** | μ = μ_max × S₁/(K₁+S₁) × S₂/(K₂+S₂) | μ_max, K₁, K₂ | Two limiting substrates |
| **Logistic** | dX/dt = μ_max·X·(1 - X/X_max) | μ_max, X_max | Self-limiting (no substrate) |

### 3.2 Substrate Uptake

**Biomass-substrate yield:**
dS/dt = -(1/Y_XS) · dX/dt - m_s·X - (1/Y_PS) · dP/dt

Where:
- Y_XS = biomass yield on substrate (g biomass / g substrate)
- m_s = maintenance coefficient (g substrate / g biomass / h)
- Y_PS = product yield on substrate

**Typical Y_XS values:**
| Organism | Substrate | Y_XS (g/g) |
|---|---|---|
| E. coli | Glucose | 0.40–0.50 |
| S. cerevisiae | Glucose (aerobic) | 0.45–0.50 |
| S. cerevisiae | Glucose (anaerobic) | 0.08–0.12 |
| B. subtilis | Glucose | 0.35–0.45 |
| P. pastoris | Glycerol | 0.40–0.55 |
| A. niger | Glucose | 0.45–0.50 |
| C. glutamicum | Glucose | 0.45–0.55 |
| Lactobacillus | Glucose | 0.15–0.25 |

### 3.3 Product Formation (Luedeking-Piret)

dP/dt = α · dX/dt + β · X

| Type | α | β | Examples |
|---|---|---|---|
| **Growth-associated** | >0 | ≈0 | Ethanol (yeast), lactate (bacteria) |
| **Non-growth-associated** | ≈0 | >0 | Antibiotics, secondary metabolites |
| **Mixed** | >0 | >0 | Amino acids, citric acid |

### 3.4 Byproduct Formation

**Overflow metabolism (Crabtree effect in yeast, acetate in E. coli):**

When q_s > q_s_crit (critical substrate uptake rate):
→ Excess carbon → overflow to byproduct

```
q_P = 0                          if q_S ≤ q_S_crit
q_P = Y_PS × (q_S - q_S_crit)   if q_S > q_S_crit
```

### 3.5 Oxygen Uptake

dO₂/dt = k_L·a × (O₂* - O₂) - q_O₂·X

**OUR (Oxygen Uptake Rate):**
q_O₂ = μ/Y_XO + m_O

**Typical k_L·a values:** 50–500 h⁻¹ (lab), 200–1000 h⁻¹ (industrial)

### 3.6 Heat Generation

Q = Y_QX · dX/dt + Y_QS · (-dS/dt)

Typical Y_QX: 10–25 kJ/g biomass

### 3.7 pH Effects

**Effect on growth rate:**
μ_eff = μ_opt × f(pH)

Where f(pH) follows a Gaussian or cardinally-shaped function:
f(pH) = exp(-((pH - pH_opt)/σ)²)

**pH changes from metabolism:**
- Glucose consumption → organic acids → pH ↓
- Amino acid consumption → NH₃ release → pH ↑
- Lactate production → pH ↓
- CO₂ production → H₂CO₃ → pH ↓ (buffered)

### 3.8 Temperature Effects

**Arrhenius relationship:**
μ(T) = A × exp(-E_a/RT)

**Cardinal temperature model:**
μ(T) = μ_opt × γ(T)

Where γ(T) is a piecewise function with T_min, T_opt, T_max.

| Organism | T_opt (°C) | T_range (°C) | pH_opt | pH_range |
|---|---|---|---|---|
| E. coli | 37 | 15–45 | 7.0 | 5.5–8.5 |
| S. cerevisiae | 30 | 5–40 | 5.0–5.5 | 3.5–7.0 |
| B. subtilis | 37 | 15–55 | 7.0 | 5.5–9.0 |
| P. pastoris | 30 | 15–35 | 5.0–6.0 | 3.0–7.0 |
| A. niger | 30–35 | 20–45 | 5.0–6.0 | 2.5–8.0 |
| C. glutamicum | 30 | 15–40 | 7.0–7.5 | 6.0–9.0 |
| Lactobacillus | 37–42 | 15–50 | 5.5–6.5 | 4.0–8.0 |
| Z. mobilis | 30 | 15–40 | 5.0–6.0 | 3.5–7.5 |
| P. putida | 30 | 10–42 | 7.0 | 5.5–8.0 |

---

## 4. Dead Cell Dynamics & Reuse

### 4.1 Cell Death
dX_d/dt = k_d · X_v

Death rate (k_d) increases at:
- Substrate depletion (starvation)
- Toxic product accumulation
- pH outside tolerable range
- Temperature stress
- Osmotic stress

### 4.2 Cell Lysis & Nutrient Recycling
Dead cells can lyse, releasing:
- Amino acids → re-consumed as nitrogen source
- Nucleic acids → phosphate source
- Cell debris → particulates

dN_recycled/dt = k_lysis · X_d

### 4.3 Viability Model
Viability = X_v / (X_v + X_d)

d(viability)/dt = f(S, P, pH, T, time)

---

## 5. Public Data & Knowledge Sources

### Primary Databases
| Source | URL | Content |
|---|---|---|
| **KEGG** | https://www.genome.jp/kegg/ | Metabolic pathways (free API) |
| **BioCyc** | https://biocyc.org/ | Metabolic pathways per organism (free tier) |
| **MetaCyc** | https://metacyc.org/ | Universal metabolic database |
| **BRENDA** | https://www.brenda-enzymes.org/ | Enzyme kinetics database |
| **SABIO-RK** | https://sabiork.h-its.org/ | Reaction kinetics data |
| **BioModels** | https://www.ebi.ac.uk/biomodels/ | Curated mathematical models (SBML) |
| **UniProt** | https://www.uniprot.org/ | Protein sequences & function annotations |

### Metabolic Models (Genome-Scale)
| Database | URL | Content |
|---|---|---|
| **BiGG Models** | http://bigg.ucsd.edu/ | GEMs for 100+ organisms (SBML/JSON) |
| **MEMOTE** | https://memote.io/ | GEM quality testing suite |
| **CarveMe** | https://github.com/cdanielmachado/carveme | Generate organism-specific GEMs |
| **ModelSEED** | https://modelseed.org/ | Automated GEM reconstruction |

### Specific Published Models (Open Access)
| Model | Organism | Key Features |
|---|---|---|
| **iJO1366** | E. coli K-12 MG1655 | 1366 genes, 2251 reactions |
| **iML1515** | E. coli BL21(DE3) | Protein production strain |
| **Yeast8** | S. cerevisiae | 3900+ reactions |
| **iBsu1103** | B. subtilis 168 | Gram-positive model |
| **iCGB21FR** | C. glutamicum ATCC 13032 | Amino acid producer |
| **iPpa1257** | P. pastoris | Methylotrophic yeast |

### Python Libraries for Metabolic Modeling
| Library | URL | Purpose |
|---|---|---|
| **COBRApy** | https://github.com/opencobra/cobrapy | FBA, FVA, gene knockouts |
| **cameo** | https://github.com/biosustain/cameo | Strain design, pathway engineering |
| **reframed** | https://github.com/cdanielmachado/reframed | Community modeling |
| **pyFOOMB** | https://github.com/JuBiotech/pyFOOMB | Dynamic bioprocess modeling |
| **scipy.integrate** | SciPy | ODE solver (solve_ivp) |

---

## 6. Bioreactor Configurations

### 6.1 Operation Modes
| Mode | Description | Mathematical Form |
|---|---|---|
| **Batch** | No inflow/outflow | dV/dt = 0 |
| **Fed-batch** | Inflow, no outflow | dV/dt = F_in |
| **Continuous (chemostat)** | Inflow = outflow | dV/dt = 0, D = F/V |
| **Perfusion** | Cell retention + media exchange | Complex cell balance |

### 6.2 Mass Balances (General)

**Biomass (viable):**
d(X_v·V)/dt = (μ - k_d)·X_v·V - F_out·X_v

**Substrate:**
d(S·V)/dt = -q_S·X_v·V + F_in·S_feed - F_out·S

**Product:**
d(P·V)/dt = q_P·X_v·V - F_out·P

**Byproduct:**
d(B·V)/dt = q_B·X_v·V - F_out·B

**Oxygen (dissolved):**
d(O₂·V)/dt = k_L·a·(O₂* - O₂)·V - q_O₂·X_v·V

**Volume:**
dV/dt = F_in - F_out - F_evap

---

## 7. Proposed Configurable Organism Schema

```yaml
organism:
  id: "ecoli_k12"
  name: "Escherichia coli K-12"
  type: bacteria  # bacteria, yeast, fungi
  gram: negative  # for antibiotics/resistance
  genome_model: "iJO1366"

  growth:
    model: monod  # monod, moser, andrews, logistic
    mu_max: 0.7   # h⁻¹
    K_s: 0.02     # g/L (glucose)
    Y_XS: 0.50    # g/g
    m_s: 0.025    # g/g/h (maintenance)

  death:
    k_d_max: 0.01  # h⁻¹
    k_lysis: 0.005  # h⁻¹

  substrate_uptake:
    substrates:
      - name: glucose
        q_S_max: 1.5    # g/g/h
        K_m: 0.01       # g/L
        priority: 1      # used first
      - name: glycerol
        q_S_max: 0.8
        K_m: 0.05
        priority: 2

  products:
    - name: acetate
      model: luedeking-piret
      alpha: 0.1     # growth-associated
      beta: 0.02     # non-growth-associated (h⁻¹)
      q_S_crit: 0.8  # overflow threshold (g/g/h)
      Y_PS: 0.15
      inhibition:
        type: aiba
        K_p: 10.0    # g/L (50% inhibition)

  oxygen:
    q_O2_max: 0.4    # g/g/h
    Y_XO: 0.95       # g biomass / g O₂
    m_O: 0.01        # g/g/h

  environment:
    temperature:
      model: cardinal
      T_min: 8
      T_opt: 37
      T_max: 45
    pH:
      pH_min: 5.0
      pH_opt: 7.0
      pH_max: 9.0
      sigma: 0.8      # tolerance width

  bioreactor_defaults:
    kLa: 150          # h⁻¹
    O2_star: 0.007    # g/L (saturation)
    heat_transfer: 0.5 # kJ/h/K
```

---

## 8. Architecture Integration with Existing Tools

```
                    ┌──────────────────────────┐
                    │   Config UI (Streamlit)   │
                    │  Organism selector, param │
                    │  editor, scenario setup   │
                    └──────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Organism DB    │  │ Simulation Engine│  │  Data I/O       │
│  (YAML/JSON)    │  │ (scipy.integrate)│  │ (pandas/csv)    │
│  - e_coli       │  │ - ODE system     │  │ - Import CSV    │
│  - s_cerevisiae │  │ - Stoichiometry  │  │ - Fit params    │
│  - p_pastoris   │  │ - Events (feeds) │  │ - Export results│
│  - ...          │  │                  │  │                  │
└─────────────────┘  └────────┬─────────┘  └─────────────────┘
                               │
          ┌────────────────────┤
          ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  COBRApy/FBA     │  │  Parameter Est.  │
│  (metabolic opt) │  │  (scipy.optimize) │
└─────────────────┘  └─────────────────┘
```

---

## 9. Next Steps

1. ✅ Analyze datasets (done)
2. ✅ Compile metabolic equations and parameters from public sources (done)
3. ⬜ Build organism configuration schema
4. ⬜ Implement ODE simulation engine
5. ⬜ Load & fit against DTB experimental data
6. ⬜ Build Streamlit UI
7. ⬜ Validate predictions against holdout data
