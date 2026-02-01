# Training Process Guide

## Setup and Execution Process

### 1. Environment Setup
```bash
bash prepare_all_science.sh
```

### 2. Start Sandbox
```bash
cd sandbox
uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4
```

### 3. Run Training Script
1. Change `PATH_PREFIX` to the ARLArena project path in the bash file 
2. Execute training:
```bash
bash examples/math_trainer/4B/train_sapo.sh
```

## Verification Checklist

Please verify the following:

- [ ] **Sandbox Status**: Sandbox returns `200 OK`
- [ ] **Rollout Data**: `ROLLOUT_DATA_DIR` contains saved `.jsonl` files

## Expected Return Files

1. **ROLLOUT_DATA_DIR folder**: Contains rollout data in JSONL format
2. **Final model checkpoint**: Saved model after training completion