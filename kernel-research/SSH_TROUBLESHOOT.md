# Lambda SSH Troubleshooting Guide

Generated: 2026-03-23 22:55 UTC

## Current Status (all branches blocked)

All agent branches lack authorized SSH key for ubuntu@192.222.55.210.

TCP connectivity: CONFIRMED (SSH banner: Ubuntu 24.04.2 LTS, OpenSSH_9.6p1)
Authentication: BLOCKED (Permission denied, publickey)

## Resolution Steps

### Step 1: If you have the original private key

The original Lambda private key was provisioned when the instance was set up.
ooo's triage (22:19 UTC) confirmed their sandbox has the key.

Run from any branch WITH the original key:

```bash
# Add the serious-inference-engineer public key
ssh -i ~/.ssh/id_ed25519 ubuntu@192.222.55.210 \
  'echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILKWTLpj6PASNUEdl6SF3/krwFGmF8rU00KnaijNkgMq serious-inference-engineer-eval-harness" >> ~/.ssh/authorized_keys'

# Verify it worked
ssh -i ~/.ssh/id_ed25519 ubuntu@192.222.55.210 'echo OK'
```

### Step 2: After adding the key, run bootstrap

From any branch:
```bash
# Test new key
ssh -i /root/.ssh/id_ed25519 ubuntu@192.222.55.210 'echo CONFIRMED'

# Bootstrap the research environment
ssh -i /root/.ssh/id_ed25519 ubuntu@192.222.55.210 \
  'git clone https://github.com/prsabahrami/voice-ai-research /home/ubuntu/voice-ai-research 2>/dev/null || git -C /home/ubuntu/voice-ai-research pull && bash /home/ubuntu/voice-ai-research/kernel-research/bootstrap.sh'
```

### Step 3: Environment inventory

```bash
ssh -i /root/.ssh/id_ed25519 ubuntu@192.222.55.210 '
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
  python3 -c "import triton; print(triton.__version__)" 2>/dev/null || echo "triton: not installed"
  nvcc --version 2>/dev/null | head -2
  pip show flash-attn 2>/dev/null | head -2
'
```

### Step 4: Run first GPU benchmarks

```bash
ssh -i /root/.ssh/id_ed25519 ubuntu@192.222.55.210 \
  'bash /home/ubuntu/kernel-research/lambda_quick_start.sh'
```

## Alternative Resolution (via Lambda Labs portal)

If no agent has the private key:
1. Log into Lambda Labs dashboard
2. Navigate to the instance (192.222.55.210)
3. Add authorized key via the console/user-data mechanism
4. Key to add:
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILKWTLpj6PASNUEdl6SF3/krwFGmF8rU00KnaijNkgMq serious-inference-engineer-eval-harness

## Contact

Email escalation sent to: zayaan@talkingcomputers.ai (22:17 UTC)
Subject: Kernel Research Sprint - Lambda SSH Access Needed

## What's Ready to Run

All code is in GitHub. Once SSH is available:

```
/home/ubuntu/voice-ai-research/kernel-research/
  lambda_quick_start.sh    <- Run this first (H18 torch.compile baseline)
  bootstrap.sh             <- Full env setup
  harness/                 <- Timing harness
  eval/eval_harness.py     <- Result certification gatekeeper
  hypotheses/gpu_kernels.py <- GPU implementations (h001-h004)
  hypotheses/hypotheses.md  <- 20 ranked hypotheses
  results/                 <- results.jsonl for writing results
```
