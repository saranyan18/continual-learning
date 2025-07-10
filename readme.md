# 🧠 OneShot_MetaReplay: Spectral Filter Utility Modulation for Continual Learning

This project implements a **One-Shot Continual Learning framework** using **Spectral Filter Flux Tracking** and **Group-wise Utility Modulation**. It combines filter-wise frequency analysis with activation-gradient-based utility to mitigate catastrophic forgetting in task-incremental learning setups.

---

## 📦 Project Structure

```
.
├── OneShot_MetaReplay
│   ├── train
│   │   ├── grp_util_tracker.py       # Group utility computation via activation x gradient
│   │   ├── overlay_mod.py            # Overlay modulation using group FIS scores
│   │   └── trainer.py                # Training loop logic
│   └── utils
│       └── utils.py                  # Filter grouping, FIS computation
├── cnn.py                            # ResNet18 backbone
├── dataset_loader.py                 # Task-incremental CIFAR-100 loader
├── grp_util_tracker.py              # (duplicate) Group utility tracker
├── overlay_mod.py                   # (duplicate) Overlay modulator
├── spectral_tracker.py              # Tracks and saves filter spectra per task
├── trainer.py                        # Main training and evaluation loop
├── utils.py                          # (duplicate) Common utility functions
```

---

## 🚀 How It Works

### 1. **Filter Spectral Flux**
   - Tracks FFT of CNN filters across tasks.
   - Computes **spectral flux** to detect significant changes.
   - Identifies filters most affected by task drift.

### 2. **Group Utility Tracker**
   - Computes utility per filter using `activation × gradient`.
   - Aggregates utility scores over pre-defined filter groups.

### 3. **Overlay Modulator**
   - Combines **spectral flux** and **utility scores**.
   - Adjusts per-group gradient scaling via dynamic overlay.
   - Promotes stability in previously useful filters.

### 4. **Continual Training**
   - Trains on 5 CIFAR-100 tasks sequentially.
   - Applies overlay modulation from **Task 2 onwards**.

---

## 🧪 Results

- Tracks per-task accuracy and final CIFAR-100 performance.
- Logs forgetting metrics and overall accuracy post all tasks.

---

## 📊 Example Output

```
🔁 Task 1
Epoch 1, Batch 0, Loss: 4.6052
...
🎯 Accuracy on Task 1: 42.31%

🔁 Task 2
Overlay modulation applied.
...
🎯 Accuracy on Task 2: 44.57%
...
📉 Final Task Accuracies:
Task 1: 38.22%
Task 2: 44.57%
...

🌍 Final Evaluation on Full CIFAR-100 Test Set:
🧠 Total CIFAR-100 Accuracy: 46.92%
```

---

## 🛠️ Setup

```bash
# Clone repo
git clone https://github.com/your_username/continual-learning.git
cd continual-learning

# Install dependencies
pip install -r requirements.txt

# Run training
python trainer.py
```

---

## 🧠 Notes

- CIFAR-100 is split into 5 incremental tasks of 20 classes each.
- Duplicate files (e.g., `grp_util_tracker.py`, `utils.py`) exist at root and in subfolders — refactor recommended.
- `logger.py` handles experiment logging (not shown here, assumed present).

---

## 📚 Citation (if you plan to publish)

If this work inspires your research or you're building upon it, consider citing once the paper is published.

---

## 🔥 TODO

- [ ] Add unit tests
- [ ] Modularize duplicate code
- [ ] Extend to more datasets (e.g., TinyImageNet)
- [ ] Implement replay buffer or dual-mode encoder

---

## 👨‍💻 Author

Built by [@saranyan18](https://github.com/saranyan18)  
```diff
+ Research focus: Spectral Filter Dynamics, Meta Replay, Continual Learning
```
