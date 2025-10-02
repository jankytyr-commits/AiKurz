# 🚀 DQN LunarLander - Kompletní dokumentace

## 📋 Obsah
1. [Přehled projektu](#přehled-projektu)
2. [Architektura](#architektura)
3. [Soubory a jejich funkce](#soubory-a-jejich-funkce)
4. [Instalace](#instalace)
5. [Použití](#použití)
6. [Pokročilé funkce](#pokročilé-funkce)
7. [Hyperparametry](#hyperparametry)
8. [Troubleshooting](#troubleshooting)

---

## 📖 Přehled projektu

Tento projekt implementuje **Double DQN s Dueling architekturou a Prioritized Experience Replay** pro řešení úlohy **LunarLander-v3** z knihovny Gymnasium.

### 🎯 Cíl
Natrénovat agenta, který dokáže bezpečně přistát lunární modul mezi dvěma vlajkami s průměrným skóre **200+**.

### 🏆 Klíčové featury
- ✅ **Double DQN** - snižuje overestimation bias
- ✅ **Dueling architecture** - oddělené value a advantage streams
- ✅ **Prioritized Experience Replay (PER)** - efektivnější učení
- ✅ **Adaptivní learning rate** s exponenciálním decay
- ✅ **Evaluační režim** - měření bez explorace
- ✅ **TensorBoard logging** - real-time monitoring
- ✅ **Automatic checkpointing** - ukládání nejlepších modelů

---

## 🏗️ Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                    DQN AGENT ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐         ┌────────────────┐             │
│  │  Environment  │◄────────┤  Policy Network│             │
│  │ (LunarLander) │         │  (Dueling DQN) │             │
│  └───────────────┘         └────────────────┘             │
│         │                           │                      │
│         │ state, reward, done       │ Q-values            │
│         ▼                           ▼                      │
│  ┌─────────────────────────────────────────┐              │
│  │     Prioritized Replay Buffer           │              │
│  │  (stores experiences with priorities)   │              │
│  └─────────────────────────────────────────┘              │
│         │                                                  │
│         │ sample batch (prioritized)                      │
│         ▼                                                  │
│  ┌────────────────┐         ┌────────────────┐           │
│  │ Policy Network │         │ Target Network │           │
│  │  (trainable)   │         │   (frozen)     │           │
│  └────────────────┘         └────────────────┘           │
│         │                           │                      │
│         └─────────┬─────────────────┘                      │
│                   │                                        │
│                   ▼                                        │
│         Double DQN Loss + TD-error                        │
│                   │                                        │
│                   ▼                                        │
│            Update priorities                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Dueling DQN architektura

```
Input State (8 features)
         │
         ▼
┌─────────────────┐
│ Shared Features │  ← 256→256 neurons
│   (2 layers)    │
└─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Value  │ │ Advantage  │
│ Stream │ │  Stream    │
│ V(s)   │ │ A(s,a)     │
└────────┘ └────────────┘
    │         │
    └────┬────┘
         ▼
    Q(s,a) = V(s) + (A(s,a) - mean(A))
```

---

## 📁 Soubory a jejich funkce

### 1️⃣ `train.py` - Hlavní tréninková smyčka

**Účel:** Kompletní pipeline pro trénování DQN agenta.

**Klíčové komponenty:**
- `DuelingQNetwork` - neuronová síť pro aproximaci Q-funkcí
- `PrioritizedReplayBuffer` - paměť pro ukládání zkušeností
- `compute_double_dqn_loss()` - výpočet Double DQN loss
- `evaluate()` - evaluace bez explorace
- `train()` - hlavní tréninková smyčka

**Vstup:**
```python
# Command line argumenty
--episodes 15000          # Počet epizod
--lr 1e-4                # Learning rate
--gamma 0.99             # Discount factor
--target_update 500      # Frekvence target network update
--eval_every 50          # Jak často evaluovat
# ... další parametry
```

**Výstup:**
- Checkpointy: `{save_prefix}_ep{N}.pth`
- Best model: `{save_prefix}_best_eval{score}.pth`
- Final model: `{save_prefix}_final.pth`
- TensorBoard logy v `{logdir}/`

**Ukládaná data v checkpointu:**
```python
{
    "policy_state_dict": ...,    # Váhy policy sítě
    "target_state_dict": ...,    # Váhy target sítě
    "optimizer_state_dict": ..., # Stav optimizeru
    "scheduler_state_dict": ..., # Stav LR scheduleru (pokud aktivní)
    "ep": 1000,                  # Číslo epizody
    "total_steps": 250000,       # Celkový počet kroků
    "avg100": 267.5,             # Průměr posledních 100 epizod
    "best_eval": 285.3           # Nejlepší eval reward
}
```

---

### 2️⃣ `visualize.py` - Vizualizace natrénovaného agenta

**Účel:** Spuštění a vizualizace policy natrénovaného agenta.

**Klíčové funkce:**
- `DuelingQNetwork` - stejná architektura jako v train.py
- `evaluate()` - spuštění epizod s/bez renderování
- Detailní statistiky (reward, délka, akce)
- Detekce úspěšných přistání

**Použití:**
```bash
# Základní vizualizace
python visualize.py --model checkpoints/dqn_lunar_best.pth --episodes 5 --render

# S detailním výpisem a zpožděním
python visualize.py --model checkpoints/dqn_lunar_ep5000.pth \
    --episodes 10 --render --delay 0.02

# Bez renderování (jen statistiky)
python visualize.py --model checkpoints/dqn_lunar_final.pth --episodes 50

# Tichý režim
python visualize.py --model checkpoints/dqn_lunar_best.pth \
    --episodes 20 --render --quiet
```

**Výstup:**
```
============================================================
Episode 1/5
============================================================
  Step   1 | Action: Fire main      | Q-values: [2.45, 3.21, 5.67, 2.89]
  Step   2 | Action: Do nothing     | Q-values: [3.12, 2.56, 4.23, 3.01]
  ...

✅ SUCCESS | Total reward:  245.67 | Steps:  89
  Action distribution: {'Do nothing': 12, 'Fire left': 25, ...}

============================================================
FINAL STATISTICS
============================================================
Episodes:          5
Successful lands:  4/5 (80.0%)
Average reward:     198.45 ± 67.32
Min reward:          45.23
Max reward:         267.89
Median reward:      215.67
Average length:      92.4 steps
============================================================
```

---

### 3️⃣ `eval_stats.py` - Evaluace všech checkpointů

**Účel:** Systematické vyhodnocení všech uložených checkpointů a vizualizace progressu tréninku.

**Funkce:**
- Načte všechny checkpointy ze složky
- Evaluuje každý model N epizodami
- Generuje CSV soubor se statistikami
- Vytvoří grafy s learning curves

**Použití:**
```bash
# Základní spuštění (edituj checkpoint_folder v kódu)
python eval_stats.py

# Výstup: eval_stats.csv + eval_stats.png
```

**Struktura CSV výstupu:**
```csv
Checkpoint,Episode,AvgReward,StdReward,MinReward,MaxReward,MedianReward,AvgLength
dqn_lunar_ep500.pth,500,180.45,45.23,-20.15,245.67,185.32,125.4
dqn_lunar_ep1000.pth,1000,235.67,38.91,120.45,298.23,240.15,98.7
...
```

**Generované grafy:**
1. **Training Performance** - reward s error bars + rolling average
2. **Reward Range** - min/max envelope

---

### 4️⃣ `run_optimized_training.py` - Helper pro spuštění tréninku

**Účel:** Zjednodušené spouštění tréninku s předpřipravenými konfiguracemi.

**Konfigurace:**
- `OPTIMIZED_CONFIG` - doporučená (15k epizod)
- `FAST_TEST_CONFIG` - rychlý test (1k epizod)
- `AGGRESSIVE_CONFIG` - maximální výkon (20k epizod)

**Použití:**
```bash
python run_optimized_training.py

# Interaktivní výběr:
# 1. OPTIMIZED (doporučeno)
# 2. FAST_TEST
# 3. AGGRESSIVE
# 4. CUSTOM
```

---

## 🛠️ Instalace

### Krok 1: Vytvoř virtuální prostředí
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Krok 2: Instaluj závislosti
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium[box2d]
pip install numpy matplotlib tensorboard tqdm
```

### Krok 3: Ověř instalaci
```bash
python -c "import gymnasium; import torch; print('OK')"
```

---

## 🎮 Použití

### 1. Základní trénink
```bash
# Rychlý test (100 epizod, ~10 minut)
python train.py --episodes 100 --eval_every 20

# Základní trénink (2000 epizod, ~2-3 hodiny)
python train.py

# Plný trénink (15000 epizod, ~12-15 hodin)
python train.py --episodes 15000 --target_update 500 --lr_decay
```

### 2. Optimalizovaný trénink
```bash
# Pomocí helper skriptu
python run_optimized_training.py
# → Vyber možnost 1 (OPTIMIZED)

# Nebo přímo:
python train.py \
    --episodes 15000 \
    --target_update 500 \
    --lr_decay \
    --lr_gamma 0.9999 \
    --scheduler_step_every 100 \
    --eps_end 0.01 \
    --eps_decay_steps 120000 \
    --eval_every 50 \
    --eval_episodes 20 \
    --save_prefix checkpoints_optimized/dqn_lunar \
    --logdir runs/lunar_dqn_optimized
```

### 3. Monitoring během tréninku

**TensorBoard:**
```bash
# V novém terminálu
tensorboard --logdir runs/lunar_dqn_optimized

# Otevři: http://localhost:6006
```

**Sledované metriky:**
- `train/episode_reward` - reward každé epizody
- `train/avg_reward_100` - klouzavý průměr (main metric!)
- `train/avg_loss` - průměrná loss
- `train/avg_q_value` - průměrné Q-values
- `train/epsilon` - aktuální ε (explorace)
- `train/episode_length` - délka epizod
- `train/learning_rate` - aktuální LR
- `eval/reward_mean` - evaluační reward (bez explorace)

### 4. Vizualizace natrénovaného agenta

```bash
# Najdi best model
ls checkpoints_optimized/dqn_lunar_best_*.pth

# Zobraz 10 her
python visualize.py \
    --model checkpoints_optimized/dqn_lunar_best_eval285.pth \
    --episodes 10 \
    --render

# S detailním výpisem
python visualize.py \
    --model checkpoints_optimized/dqn_lunar_best_eval285.pth \
    --episodes 5 \
    --render \
    --delay 0.02
```

### 5. Evaluace všech checkpointů

```bash
# 1. Uprav checkpoint_folder v eval_stats.py:
# checkpoint_folder = "checkpoints_optimized"

# 2. Spusť evaluaci
python eval_stats.py

# 3. Výsledky:
# - eval_stats.csv (tabulka)
# - eval_stats.png (grafy)
```

---

## 🔬 Pokročilé funkce

### Resume tréninku z checkpointu

**Postup:**
1. Najdi checkpoint: `checkpoints/dqn_lunar_ep5000.pth`
2. Uprav `train.py` - přidej funkci `load_checkpoint()`
3. Nebo zkopíruj model a pokračuj s jiným `save_prefix`

```python
# Přidat do train.py před hlavní smyčku:
if args.resume:
    checkpoint = torch.load(args.resume)
    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    target_net.load_state_dict(checkpoint["target_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_episode = checkpoint["ep"] + 1
    total_steps = checkpoint["total_steps"]
    print(f"Resumed from episode {start_episode}")
```

### Experimenty s hyperparametry

**Learning rate:**
```bash
# Nižší LR (konzervativnější)
python train.py --lr 5e-5 --episodes 20000

# Vyšší LR (rychlejší, ale rizikovější)
python train.py --lr 2e-4 --episodes 10000
```

**Target network update:**
```bash
# Častější update (rychlejší adaptace)
python train.py --target_update 250

# Méně časté (stabilnější)
python train.py --target_update 2000
```

**Epsilon decay:**
```bash
# Rychlejší decay (dřívější exploitace)
python train.py --eps_decay_steps 50000

# Pomalejší decay (delší explorace)
python train.py --eps_decay_steps 200000
```

**Větší síť:**
```bash
python train.py --hidden 512 --lr 5e-5
```

### Srovnání běhů v TensorBoard

```bash
# Spusť více tréninků s různými logdiry:
python train.py --logdir runs/run1_baseline
python train.py --lr 5e-5 --logdir runs/run2_lower_lr
python train.py --target_update 250 --logdir runs/run3_fast_target

# Zobraz všechny najednou:
tensorboard --logdir runs
```

---

## ⚙️ Hyperparametry

### Kritické parametry (nejvíce ovlivňují výkon)

| Parametr | Default | Doporučeno | Vliv |
|----------|---------|------------|------|
| `--lr` | 1e-4 | 1e-4 až 5e-5 | Rychlost učení |
| `--target_update` | 1000 | 500-1000 | Stabilita Q-values |
| `--eps_decay_steps` | 100k | 100k-150k | Délka explorace |
| `--gamma` | 0.99 | 0.99 | Diskontování |

### Explorace

| Parametr | Default | Popis |
|----------|---------|-------|
| `--eps_start` | 1.0 | Počáteční ε (100% náhodné akce) |
| `--eps_end` | 0.05 | Finální ε (5% náhodné akce) |
| `--eps_decay_steps` | 100k | Kroků do plné exploitace |

**Doporučení:**
- Pro složité prostředí: `--eps_end 0.01` (méně náhody)
- Pro jednoduché: `--eps_end 0.1` (víc explorace)

### Replay buffer

| Parametr | Default | Popis |
|----------|---------|-------|
| `--buffer_size` | 200k | Kapacita bufferu |
| `--batch_size` | 128 | Velikost batch |
| `--per_alpha` | 0.6 | Prioritizace (0=uniform, 1=full) |
| `--per_beta_start` | 0.4 | Importance sampling korekce |

### Learning rate scheduling

| Parametr | Default | Popis |
|----------|---------|-------|
| `--lr_decay` | False | Aktivace decay |
| `--lr_gamma` | 0.9999 | Multiplikátor na step |
| `--scheduler_step_every` | 100 | Kroků mezi updaty |

**Formula:** `new_lr = current_lr * (gamma ** steps)`

### Síť

| Parametr | Default | Popis |
|----------|---------|-------|
| `--hidden` | 256 | Neurons v hidden layers |

**Varianty:**
- Malá: `--hidden 128` (rychlé, méně kapacity)
- Default: `--hidden 256` (vyvážené)
- Velká: `--hidden 512` (pomalé, více kapacity)

---

## 🐛 Troubleshooting

### Problém 1: Loss exploduje (NaN)
**Příznaky:** Loss skočí na `inf` nebo `nan`

**Řešení:**
```bash
# Snížit learning rate
python train.py --lr 5e-5

# Zpřísnit gradient clipping
python train.py --max_grad_norm 1.0

# Častější target update
python train.py --target_update 500
```

### Problém 2: Agent se nenaučí (reward stagnuje)
**Příznaky:** Avg100 zůstává pod 0 po 1000+ epizodách

**Řešení:**
```bash
# Delší explorace
python train.py --eps_decay_steps 200000

# Větší buffer
python train.py --buffer_size 500000

# Vyšší learning rate
python train.py --lr 2e-4
```

### Problém 3: Nestabilní učení (velké výkyvy)
**Příznaky:** Reward skáče nahoru/dolů (+200 → -50 → +150)

**Řešení:**
```bash
# Aktivovat LR decay
python train.py --lr_decay --lr_gamma 0.9999

# Méně časté target update
python train.py --target_update 2000

# Menší batch
python train.py --batch_size 64
```

### Problém 4: Catastrophic forgetting
**Příznaky:** Po dobrém výkonu (250+) náhle pokles (<100)

**Řešení:**
```bash
# Snížit epsilon end
python train.py --eps_end 0.01

# Aktivovat LR decay
python train.py --lr_decay

# Menší LR v pozdních fázích
python train.py --lr 5e-5 --lr_decay --lr_gamma 0.99999
```

### Problém 5: TensorBoard neukazuje data
**Řešení:**
```bash
# Zkontroluj složku
ls runs/lunar_dqn_optimized

# Restartuj TensorBoard
# CTRL+C a znovu:
tensorboard --logdir runs/lunar_dqn_optimized --reload_interval 5
```

### Problém 6: CUDA out of memory
**Řešení:**
```bash
# Menší batch
python train.py --batch_size 64

# Nebo použij CPU
python train.py --force_cpu
```

### Problém 7: Visualize.py nefunguje
**Chyba:** `RuntimeError: Error(s) in loading state_dict`

**Řešení:**
```python
# Zkontroluj že DuelingQNetwork v visualize.py
# odpovídá přesně té v train.py (stejné hidden layers)
```

---

## 📊 Očekávané výsledky

### Timeline učení (optimalizovaná konfigurace)

| Epizoda | Očekávaný Avg100 | Popis |
|---------|------------------|-------|
| 0-500 | -100 → 0 | Agent se učí základy |
| 500-1500 | 0 → 150 | Postupné zlepšování |
| 1500-3000 | 150 → 200 | Dosažení target |
| 3000-8000 | 200 → 260 | Stabilizace |
| 8000-15000 | 260 → 280+ | Fine-tuning |

### Kvalitativní milníky

**Episode ~300:** Agent přestává okamžitě havarovat
**Episode ~800:** První úspěšná přistání
**Episode ~1500:** Konzistentně mezi vlajkami
**Episode ~3000:** Hladké přistávání s kontrolou
**Episode ~8000+:** Near-optimal policy

---

## 📚 Reference a zdroje

### Papery
- [DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Double DQN (van Hasselt et al., 2015)](https://arxiv.org/abs/1509.06461)
- [Dueling DQN (Wang et al., 2016)](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay (Schaul et al., 2015)](https://arxiv.org/abs/1511.05952)

### Dokumentace
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/docs/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

### Prostředí
- [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

---

## 🤝 Contributing

Pro vylepšení nebo reporting bugů:
1. Experimentuj s hyperparametry
2. Zaznamenej výsledky do TensorBoard
3. Zdokumentuj změny

---

## 📝 Licence

Tento projekt je vytvořen pro vzdělávací účely.

---

**Verze dokumentace:** 1.0  
**Datum:** 2025  
**Kompatibilita:** Python 3.8+, PyTorch 2.0+, Gymnasium 0.29+

---

## 🎓 Tips & Tricks

### 1. Rychlé testování změn
```bash
# Vždy nejprve otestuj na 100 epizodách
python train.py --episodes 100 --save_prefix test/dqn --logdir runs/test
```

### 2. Srovnání s baseline
```bash
# Vždy měj baseline běh pro srovnání
python train.py --logdir runs/baseline --seed 42
python train.py --lr 5e-5 --logdir runs/experiment1 --seed 42
```

### 3. Grid search hyperparametrů
```bash
for lr in 1e-4 5e-5 2e-4; do
    python train.py --lr $lr --logdir runs/lr_$lr --seed 42
done
```

### 4. Sledování GPU
```bash
# V novém terminálu
watch -n 1 nvidia-smi
```

### 5. Background trénink
```bash
# Linux/Mac
nohup python train.py --episodes 15000 > training.log 2>&1 &

# Sledování logu
tail -f training.log
```

---

**Hodně štěstí s tréninkem! 🚀🌙**