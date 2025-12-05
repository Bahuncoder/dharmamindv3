
🕉️ Model Training Execution Checklist
======================================

PRE-TRAINING SETUP
------------------
[ ] 1. Verify GPU availability
   Command: nvidia-smi
   
[ ] 2. Check disk space (need 50GB+)
   Command: df -h
   
[ ] 3. Verify dependencies installed
   Command: pip list | grep -E "torch|transformers|tokenizers"
   
[ ] 4. Collect training data
   - Minimum 10K texts
   - Mix of Sanskrit and English
   - Validated dharmic content
   
[ ] 5. Prepare validation set (10% of data)

TOKENIZER TRAINING
------------------
[ ] 1. Train BPE tokenizer
   Command: python model/train_dharmic_tokenizer.py --model bpe
   Expected: 30-60 minutes
   Output: trained_tokenizers/dharmic_bpe.json
   
[ ] 2. Train WordPiece tokenizer
   Command: python model/train_dharmic_tokenizer.py --model wordpiece
   Expected: 30-60 minutes
   Output: trained_tokenizers/dharmic_wordpiece.json
   
[ ] 3. Validate tokenizer quality
   - Test on sample texts
   - Check vocab size (target: 50,000)
   - Verify Sanskrit handling
   
[ ] 4. Save tokenizer to model directory
   Command: cp trained_tokenizers/* model/tokenizer/

MODEL TRAINING
--------------
[ ] 1. Configure training parameters
   File: dharmallm_config.py
   - Set batch size (start with 4-8)
   - Set learning rate (3e-5)
   - Set epochs (start with 3)
   - Enable mixed precision
   
[ ] 2. Start training
   Command: python dharmallm_training.py --full-training
   Expected: 8-12 hours
   Monitor: training loss, validation loss
   
[ ] 3. Save checkpoints
   Frequency: Every epoch
   Location: checkpoints/dharma_model_epoch_N/
   
[ ] 4. Monitor training
   - Check tensorboard: tensorboard --logdir=runs
   - Watch GPU memory
   - Monitor loss curves
   
[ ] 5. Validate model quality
   - Test on validation set
   - Check perplexity (target: < 50)
   - Test generation quality

POST-TRAINING
-------------
[ ] 1. Save final model
   Location: model/dharmallm-v1/
   Files: pytorch_model.bin, config.json, tokenizer files
   
[ ] 2. Test inference
   Command: python phase4/inference/simple_demo.py
   Expected: < 2 sec response time
   
[ ] 3. Run quality tests
   - Dharmic accuracy test
   - Sanskrit generation test
   - Response relevance test
   
[ ] 4. Create model card
   - Document architecture
   - List training data
   - Report metrics
   - Add usage examples
   
[ ] 5. Package for deployment
   - Create Docker image
   - Test containerized version
   - Document deployment steps

OPTIMIZATION (Optional)
-----------------------
[ ] 1. Model quantization
   - Convert to INT8
   - Test accuracy impact
   - Measure speed improvement
   
[ ] 2. Model compression
   - Apply pruning
   - Knowledge distillation
   - Test performance
   
[ ] 3. Deployment optimization
   - Setup model serving
   - Implement batching
   - Add caching layer

TROUBLESHOOTING
---------------
Issue: Out of memory
Solution: Reduce batch size, enable gradient accumulation

Issue: Loss not decreasing
Solution: Adjust learning rate, check data quality

Issue: Model divergence
Solution: Lower learning rate, add gradient clipping

Issue: Slow training
Solution: Enable mixed precision, optimize data loading

MONITORING COMMANDS
-------------------
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f training.log

# Check disk space
df -h /path/to/checkpoints

# View tensorboard
tensorboard --logdir=runs --port=6006

VALIDATION METRICS
------------------
Target Metrics:
- Training loss: < 2.0
- Validation loss: < 2.5
- Perplexity: < 50
- Dharmic accuracy: > 90%
- Response time: < 2 sec
- Memory usage: < 16GB

SUCCESS CRITERIA
----------------
✅ Tokenizer trained with 50K+ vocab
✅ Model trained for 3+ epochs
✅ Validation loss converged
✅ Quality tests passing
✅ Inference working correctly
✅ Checkpoints saved
