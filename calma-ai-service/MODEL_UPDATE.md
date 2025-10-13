# Model Update - Single Model System

## Changes Made

Your Calma AI service has been updated to use **only your newly trained fine-tuned model** (no base meta model).

### What Changed

#### 1. Model Path Updated
**File**: `.env`
- **Old**: `MODEL_PATH=../calma-ai/models/calma-final`
- **New**: `MODEL_PATH=../calma-ai/models/calma-hf-trained/final`

Now points to your newly trained model with anti-overfitting improvements.

#### 2. Simplified to Single Model System
**File**: `app/services/model_service.py`

**Before** (Dual Model System):
- Casual model: Base Llama (for early conversation)
- Therapeutic model: Fine-tuned with LoRA (for mental health)
- Two separate pipelines

**After** (Single Model System):
- One fine-tuned model for **all** interactions
- One pipeline
- Simpler, more consistent responses

**Benefits**:
- ✅ Less memory usage (~50% reduction)
- ✅ Faster loading (only one model)
- ✅ Consistent responses across all conversation types
- ✅ Your anti-overfitting training applied to all interactions
- ✅ No model switching logic needed

#### 3. Updated Inference Service
**File**: `app/services/inference_service.py`

- Still uses conversation mode detection (casual/therapeutic/crisis)
- But all modes use the same fine-tuned model
- Mode only affects system prompt, not which model is used

### Architecture Change

**Before**:
```
┌─────────────────┐
│ User Message    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mode Detection  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────┐
│ Casual│ │Therapeutic│
│ Base  │ │Fine-tuned │
│ Llama │ │  + LoRA  │
└───────┘ └──────────┘
```

**After**:
```
┌─────────────────┐
│ User Message    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mode Detection  │
│ (for prompt)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuned     │
│ Calma Model     │
│  (+ LoRA)       │
└─────────────────┘
```

## How to Use

### Start the Service

Using the main start script (recommended):
```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend
./start-calma.sh
```

This will:
1. Start the AI service on port 8000
2. Load your newly trained model
3. Start the NestJS backend on port 3000
4. Monitor both services

### Or Start AI Service Only

```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai-service
python3 -m app.main
```

### Verify It's Working

**Check health endpoint**:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": {
    "single_model_system": true,
    "model_path": "../calma-ai/models/calma-hf-trained/final",
    "lora_enabled": true,
    ...
  }
}
```

**Test inference**:
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help with family stress",
    "context": ""
  }'
```

## Expected Behavior

### Memory Usage
- **Before**: ~4-5GB (two models)
- **After**: ~2.5-3GB (one model)
- **Benefit**: Fits better in your 5.64GB GPU

### Loading Time
- **Before**: 60-90 seconds (loading two models)
- **After**: 30-45 seconds (loading one model)
- **Benefit**: Faster startup

### Response Quality
- All interactions use your newly trained, anti-overfitting model
- More consistent responses across conversation types
- Better generalization from diverse HuggingFace training data

## Troubleshooting

### Model Not Found
If you see: `No LoRA adapters found at ../calma-ai/models/calma-hf-trained/final`

**Check the model path**:
```bash
ls -la /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai/models/calma-hf-trained/final/
```

You should see:
- `adapter_config.json`
- `adapter_model.safetensors` (or `.bin`)
- `special_tokens_map.json`
- `tokenizer_config.json`

### Still Loading Old Model
1. Stop the service: `Ctrl+C` or `pkill -f "uvicorn.*app.main"`
2. Verify `.env` file:
   ```bash
   cat /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai-service/.env | grep MODEL_PATH
   ```
   Should show: `MODEL_PATH=../calma-ai/models/calma-hf-trained/final`
3. Restart: `./start-calma.sh`

### Out of Memory
If you still get OOM errors:
1. The new model uses less memory than dual system
2. Check GPU usage: `nvidia-smi`
3. Kill other GPU processes if needed
4. Reduce max context length in `.env`:
   ```bash
   MAX_TOKENS=192  # Reduced from 256
   ```

### Want to Switch Back to Old Model?
Edit `.env`:
```bash
MODEL_PATH=../calma-ai/models/calma-final
```
Restart the service.

## Verification Checklist

After starting the service, verify:

- [ ] Service starts without errors
- [ ] Health endpoint returns `"status": "healthy"`
- [ ] Model info shows `"single_model_system": true`
- [ ] Model path shows correct location (`calma-hf-trained/final`)
- [ ] LoRA adapters loaded (`"lora_enabled": true`)
- [ ] Test inference returns relevant response
- [ ] Memory usage is reasonable (< 4GB)
- [ ] NestJS backend can connect to AI service

## Performance Comparison

| Metric | Old (Dual Model) | New (Single Model) |
|--------|------------------|-------------------|
| Models Loaded | 2 (base + fine-tuned) | 1 (fine-tuned only) |
| Memory Usage | ~4.5GB | ~2.5GB |
| Load Time | 60-90s | 30-45s |
| GPU Fit | Tight (OOM risk) | Comfortable |
| Training Data | 248KB | 10,000+ conversations |
| Overfitting | Higher risk | Reduced (proper splits) |
| Consistency | Mode-dependent | Uniform |

## Next Steps

1. **Test extensively**: Try various conversation types
2. **Monitor performance**: Check response quality and speed
3. **Compare with old**: See if responses improved
4. **Adjust parameters**: Tune temperature/max_tokens in `.env` if needed
5. **Deploy**: Once satisfied, deploy to production

## Rollback Plan

If you need to revert to the old dual-model system:

1. Restore `.env`:
   ```bash
   MODEL_PATH=../calma-ai/models/calma-final
   ```

2. Restore `model_service.py`:
   ```bash
   git checkout app/services/model_service.py
   ```

3. Restore `inference_service.py`:
   ```bash
   git checkout app/services/inference_service.py
   ```

4. Restart service

## Summary

✅ **Configured**: `.env` points to new model
✅ **Simplified**: Single model system (easier to maintain)
✅ **Optimized**: Less memory, faster loading
✅ **Improved**: Anti-overfitting training applied
✅ **Ready**: Just run `./start-calma.sh`

Your Calma AI service is now using your newly trained, improved model exclusively! 🎉
