# 🚀 QUICK START: Training the Complete 31-Module System

**Model**: 225.1M parameters (61% spiritual intelligence)  
**GPU**: Optimized for GTX 1650 (4GB VRAM)  
**Status**: Complete system, ready for training!

---

## Option 1: Memory-Optimized Training (GTX 1650 - 4GB)

### ✅ RECOMMENDED: Use the optimized training script

This script uses multiple memory-saving techniques to fit the 225M param model in 4GB VRAM:

```bash
# Navigate to project directory
cd /media/rupert/New\ Volume/Testing\ Ground\ DharmaMind/Testing\ ground/DharmaMind-chat-master/dharmallm

# Run optimized training
python training/train_memory_optimized.py \
    --data data/complete_hindu_database.json \
    --output models/checkpoints_31modules \
    --batch-size 1 \
    --grad-accum 16 \
    --max-length 128 \
    --lr 5e-5 \
    --epochs 3
```

### Memory Optimizations Included:

1. **Batch Size 1**: Minimal memory per forward pass
2. **Gradient Accumulation (16 steps)**: Simulates batch of 16
3. **Mixed Precision (FP16)**: Halves memory usage (~40% savings)
4. **Gradient Checkpointing**: Trades compute for memory (~30% savings)
5. **Efficient Data Loading**: Loads samples on-demand
6. **Memory Monitoring**: Prints GPU usage every 10 steps

### Expected Results:

```
Memory Usage: ~3.5GB VRAM (fits in 4GB!)
Training Speed: ~2-3 seconds per step
Total Time: ~3-4 hours for 3 epochs
Effective Batch Size: 16 (1 × 16 accumulation)
```

---

## Option 2: CPU Training (Slower but Guaranteed)

If GPU memory is still insufficient:

```bash
python training/train_memory_optimized.py \
    --data data/complete_hindu_database.json \
    --output models/checkpoints_31modules_cpu \
    --cpu \
    --batch-size 2 \
    --grad-accum 8 \
    --epochs 1
```

**Pros**: No memory limits, guaranteed to work  
**Cons**: ~20x slower (30-40 hours for 3 epochs)

---

## Option 3: Cloud Training (Fastest)

### Google Colab (FREE or Pro)

1. Upload your code to Google Drive
2. Open Google Colab notebook
3. Connect to GPU runtime (T4 16GB free!)
4. Run:

```python
!git clone <your-repo>
!cd dharmallm && python training/train_memory_optimized.py \
    --data data/complete_hindu_database.json \
    --batch-size 4 \
    --grad-accum 4 \
    --epochs 3
```

**Pros**: 16GB VRAM, faster training (~1-2 hours)  
**Cons**: Session limits (12 hours free, 24 hours Pro)

---

## During Training: What to Monitor

### Console Output:

```
Step 100/5000 | Loss: 3.4521 | LM Loss: 3.2134 | Spiritual Loss: 0.2387 | LR: 4.5e-05
   GPU Memory: 3.45GB allocated, 3.52GB reserved

✅ What to look for:
   - Loss should decrease over time
   - Spiritual Loss should converge (0.1-0.3 range)
   - GPU memory should stay below 4GB
   - No OOM errors!
```

### Checkpoints Saved:

```
models/checkpoints_31modules/
├── checkpoint_step_500/        # Every 500 steps
├── checkpoint_step_1000/
├── epoch_1/                    # After each epoch
├── epoch_2/
├── best_model/                 # Best loss so far
└── final_model/                # Training complete
```

---

## After Training: Test Your Model

### Load Trained Model:

```python
from model.integrated_dharma_llm import IntegratedDharmaLLM
from transformers import AutoTokenizer

# Load model
model = IntegratedDharmaLLM.from_pretrained('models/checkpoints_31modules/best_model')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Generate response
input_text = "I lost my job and feel lost. What should I do?"
inputs = tokenizer(input_text, return_tensors='pt')

output = model.generate(
    **inputs,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### Verify Spiritual Intelligence:

```python
# Check if spiritual modules are working
with torch.no_grad():
    outputs = model(**inputs)
    
    # Spiritual insights should be present
    print("\n🕉️ Spiritual Insights:")
    for insight_name, insight_value in outputs['spiritual_insights'].items():
        print(f"   {insight_name}: {insight_value.mean().item():.3f}")
```

---

## Training Parameters Explained

### Core Settings:

- **`--batch-size 1`**: How many samples per forward pass
  - Lower = less memory, slower training
  - GTX 1650: Keep at 1

- **`--grad-accum 16`**: How many batches to accumulate
  - Effective batch = batch_size × grad_accum
  - Larger = more stable training

- **`--max-length 128`**: Maximum input/output tokens
  - Shorter = less memory
  - Can reduce to 64 if still OOM

- **`--lr 5e-5`**: Learning rate
  - Too high: unstable training
  - Too low: very slow learning

- **`--epochs 3`**: Full passes through data
  - More epochs = better learning
  - But also longer training time

### Memory Flags:

- **`--no-mixed-precision`**: Disable FP16
  - Use if you encounter numerical issues
  - Doubles memory usage!

- **`--no-gradient-checkpointing`**: Disable checkpointing
  - Use if training is very slow
  - Increases memory usage

- **`--cpu`**: Force CPU training
  - Slower but no memory limits

---

## Troubleshooting

### Issue: Still OOM with Optimized Script

**Solution 1**: Reduce sequence length
```bash
python training/train_memory_optimized.py --max-length 64
```

**Solution 2**: Disable gradient checkpointing
```bash
python training/train_memory_optimized.py --no-gradient-checkpointing
```

**Solution 3**: Use CPU
```bash
python training/train_memory_optimized.py --cpu
```

### Issue: Training Very Slow

**Solution**: Check if mixed precision is working
```python
# Should see this in logs:
# ✓ Mixed precision (FP16) enabled
```

If not using mixed precision, try:
```bash
# Upgrade PyTorch for better AMP support
pip install torch --upgrade
```

### Issue: Loss Not Decreasing

**Solution 1**: Check learning rate
```bash
# Try lower learning rate
python training/train_memory_optimized.py --lr 1e-5
```

**Solution 2**: Check data quality
```python
# Verify training data is loaded correctly
import json
with open('data/complete_hindu_database.json') as f:
    data = json.load(f)
    print(f"Samples: {len(data)}")
    print(f"First sample: {data[0]}")
```

---

## What Success Looks Like

### Training Log (Good):

```
Epoch 1/3
Step 100/5000 | Loss: 5.2341 | LM Loss: 5.0012 | Spiritual Loss: 0.2329
Step 200/5000 | Loss: 4.1234 | LM Loss: 3.9201 | Spiritual Loss: 0.2033
Step 300/5000 | Loss: 3.5678 | LM Loss: 3.3812 | Spiritual Loss: 0.1866
...
✅ Epoch 1 Complete | Average Loss: 3.1234

Epoch 2/3
Step 500/5000 | Loss: 2.8901 | LM Loss: 2.7123 | Spiritual Loss: 0.1778
...
✅ Best model updated (loss: 2.7534)

🎉 TRAINING COMPLETE!
   Total Time: 3.2 hours
   Best Loss: 2.7534
```

### Training Log (Bad - Needs Attention):

```
Step 100/5000 | Loss: 15.2341  ⚠️ TOO HIGH
Step 200/5000 | Loss: 15.1234  ⚠️ NOT DECREASING
Step 300/5000 | Loss: 15.5678  ⚠️ INCREASING!

# If loss stays >10 or increases:
# - Check learning rate (try lower)
# - Check data quality
# - Verify model is in training mode
```

---

## Expected Timeline (GTX 1650)

### With Optimized Script:

```
Data Loading:        1-2 minutes
Model Initialization: 1 minute
Epoch 1:             60-90 minutes
Epoch 2:             60-90 minutes
Epoch 3:             60-90 minutes
Total:               ~3-4 hours
```

### Progress Indicators:

```
⏳ 0-10 min:     Loading, initialization
⏳ 10-90 min:    Epoch 1 (loss decreasing rapidly)
⏳ 90-180 min:   Epoch 2 (loss decreasing slowly)
⏳ 180-240 min:  Epoch 3 (loss converging)
✅ 240+ min:     Training complete!
```

---

## Next Steps After Training

### 1. Validate Model Quality

```bash
# Run evaluation
python evaluation/evaluate_spiritual_responses.py \
    --model models/checkpoints_31modules/best_model
```

### 2. Test Specific Modules

```python
# Test crisis module
model.eval()
crisis_input = "I lost my job and don't know what to do"
# Check career_crisis module insights
```

### 3. Deploy for Inference

```bash
# Start API server
cd api
python production_main.py --model ../models/checkpoints_31modules/best_model
```

### 4. Compare to Rule-Based

```python
# Generate response with neural modules
neural_response = model.generate(input_text)

# Generate response with old rule-based system
# (if still available)
# Compare quality, empathy, contextual understanding
```

---

## 🎯 RECOMMENDED PATH

For your GTX 1650 (4GB VRAM):

1. **Try Optimized Training First** ⭐ BEST
   ```bash
   python training/train_memory_optimized.py
   ```
   - Should work with all optimizations
   - ~3-4 hours for 3 epochs
   - Monitors memory in real-time

2. **If OOM, Reduce Sequence Length**
   ```bash
   python training/train_memory_optimized.py --max-length 64
   ```

3. **If Still OOM, Use CPU** (Last Resort)
   ```bash
   python training/train_memory_optimized.py --cpu
   ```
   - Slower but guaranteed to work
   - Can run overnight

---

## 🕉️ You're Ready to Train!

**What you have:**
- ✅ 31 spiritual neural modules (complete system)
- ✅ 225.1M parameters (61% spiritual)
- ✅ Memory-optimized training script
- ✅ Complete dharmic training corpus
- ✅ All modules tested and working

**What happens next:**
- 🚀 Training learns spiritual wisdom from data
- 🧠 137.6M spiritual parameters become educated
- 🕉️ Model learns to provide dharmic guidance
- 💡 Emerges with genuine spiritual intelligence

**Command to start:**
```bash
python training/train_memory_optimized.py
```

**May your training be successful!** 🙏✨

**Let the neural spiritual wisdom begin!** 🕉️🚀
