# 🚀 Improved Training V2 - Quick Start Guide

## What's Different?

### V1 (Previous - Poor Results)
- 20 epochs only
- Learning rate: 0.0003 (too high)
- No validation set
- Saved "best" at epoch 0
- No gradient clipping
- No early stopping

### V2 (Improved - Better Quality)
- **100 epochs** (with early stopping)
- **Learning rate: 0.0001** (more stable)
- **10% validation set** for monitoring
- **Saves truly best model** based on validation loss
- **Gradient clipping** (prevents explosions)
- **Early stopping** (stops if no improvement)
- **Regular checkpoints** (every 5 epochs)
- **Better learning rate schedule** (CosineAnnealingWarmRestarts)

## Key Improvements

### 1. Lower Learning Rate
```
V1: 0.0003 → Unstable, model diverged
V2: 0.0001 → Stable, better convergence
```

### 2. More Epochs
```
V1: 20 epochs → Not enough training
V2: 100 epochs → Proper training (will stop early if converged)
```

### 3. Validation Monitoring
```
V1: No validation → Saved worst model as "best"
V2: 10% validation → Saves truly best model
```

### 4. Gradient Clipping
```
V1: No clipping → Gradients explode
V2: Clip at 1.0 → Stable gradients
```

### 5. Early Stopping
```
V1: Always runs 20 epochs
V2: Stops if no improvement for 15 epochs (saves time)
```

## Configuration

```python
config = {
    'epochs': 100,              # Maximum epochs
    'batch_size': 16,           # Batch size (same as V1)
    'learning_rate': 0.0001,    # Lower for stability
    'weight_decay': 0.01,       # L2 regularization
    'warmup_epochs': 5,         # Warmup period
    'grad_clip': 1.0,          # Gradient clipping
    'patience': 15,            # Early stopping patience
    'val_split': 0.1,          # 10% validation
    'checkpoint_every': 5,     # Save every 5 epochs
    'log_every': 25,           # Log every 25 batches
}
```

## How to Run

### Option 1: Quick Start (Recommended for overnight)
```bash
./train.sh
```

### Option 2: Background with logging
```bash
nohup ./train.sh > training_v2.log 2>&1 &
```

### Option 3: With custom config (advanced)
```bash
python3 scripts/training/train_improved_v2.py
```

## Expected Results

### Training Time
- **Per Epoch:** ~2 minutes (same as before)
- **Total:** 2-6 hours depending on early stopping
  - If converges at epoch 30: ~1 hour
  - If converges at epoch 60: ~2 hours
  - Full 100 epochs: ~3.5 hours

### Model Quality
Expected scores after proper training:
- **Sanskrit Content:** 80-95% (vs 3-8% in V1)
- **Quality Score:** 70-90/100 (vs 6.7/100 in V1)
- **Coherence:** Much better
- **Repetition:** Minimal

## Monitoring

### During Training
```bash
# Watch training progress
tail -f logs/training/training_log_v2.txt

# Or use the monitor
./monitor.sh
```

### Check Progress
```bash
# See latest checkpoint
ls -lht model/checkpoints/ | head -5

# Check validation loss trend
grep "Val Loss" logs/training/training_log_v2.txt
```

## Checkpoints

### What Gets Saved

1. **best_model.pt** - Best validation loss (THIS IS THE ONE TO USE!)
2. **checkpoint_epoch_5.pt** - Regular checkpoint at epoch 5
3. **checkpoint_epoch_10.pt** - Regular checkpoint at epoch 10
4. ... etc every 5 epochs

### How to Test After Training
```bash
# Test the best model
python3 scripts/training/test_master_model.py

# Or test specific checkpoint
python3 scripts/training/test_master_model.py --checkpoint model/checkpoints/checkpoint_epoch_50.pt
```

## Troubleshooting

### If Training is Too Slow
- Reduce batch_size to 8
- Reduce epochs to 50
- The model will still work, just may be slightly less accurate

### If Out of Memory
```python
# Edit train_improved_v2.py, change:
'batch_size': 8,  # Instead of 16
```

### If Loss Not Decreasing
- Let it run longer (sometimes takes 20-30 epochs to see improvement)
- Check if learning rate is too low/high
- Verify data is loading correctly

### If Early Stopping Too Soon
```python
# Edit train_improved_v2.py, change:
'patience': 25,  # Instead of 15
```

## Expected Output Pattern

```
Epoch 1: Train Loss: 3.85 | Val Loss: 3.90
Epoch 5: Train Loss: 3.45 | Val Loss: 3.52
Epoch 10: Train Loss: 3.15 | Val Loss: 3.25 ← Improving!
Epoch 20: Train Loss: 2.85 | Val Loss: 2.95 ← Good progress
Epoch 30: Train Loss: 2.65 | Val Loss: 2.72 ← Best so far!
Epoch 40: Train Loss: 2.55 | Val Loss: 2.75 ← Slight overfit
Epoch 50: Train Loss: 2.48 | Val Loss: 2.73 ← No improvement
...
Epoch 60: Early stopping triggered (no improvement for 15 epochs)
Best model: Epoch 30 with Val Loss: 2.72
```

## Success Criteria

### Good Training Run:
- ✅ Validation loss decreases steadily
- ✅ Train/Val gap < 0.3 (not overfitting)
- ✅ Early stopping between epochs 30-70
- ✅ Final val loss < 2.5

### Poor Training Run:
- ❌ Validation loss increases
- ❌ Train/Val gap > 0.5 (overfitting)
- ❌ Loss stays flat after epoch 10
- ❌ NaN or Inf losses

## After Training

Once training completes with good results:

1. **Test the model**
   ```bash
   python3 scripts/training/test_master_model.py
   ```

2. **Check quality scores**
   - Should be 70-90/100
   - Sanskrit content should be 80%+

3. **Deploy to API**
   ```bash
   cd api
   python3 main.py
   ```

4. **Integration with DharmaLLM**
   - Connect to RAG system
   - Test with spiritual intelligence modules
   - User testing

## Estimated Timeline

- **Tonight:** Start training (3-6 hours)
- **Tomorrow Morning:** Test results
- **Tomorrow Afternoon:** Deploy if good, retrain if needed
- **This Week:** Production integration

---

**Ready to start?**

```bash
# Make sure you're in the project directory
cd "/media/rupert/New Volume/Dharmamind/FinalTesting/dharmallm"

# Start training
./train.sh

# Or in background
nohup ./train.sh > training_v2.log 2>&1 &

# Monitor progress
tail -f logs/training/training_log_v2.txt
```

Good luck! 🕉️
