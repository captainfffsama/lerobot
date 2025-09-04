# how to change sheduler
```bash
"--scheduler.type=periodic_cosine_with_decay_peaks",
"--scheduler.num_cycles=7",
"--scheduler.num_warmup_steps=1000"
"--scheduler.cycle_length=10000",
"--scheduler.initial_peak_lr=1e-4",
"--scheduler.final_peak_lr=5e-6",
"--scheduler.min_lr=2.5e-6"
"--optimizer.type=adamw",
"--optimizer.betas=[0.9,0.95]",
"--optimizer.eps=1e-8",
"--optimizer.grad_clip_norm=10.0",
"--optimizer.lr=0.0001",
"--optimizer.weight_decay=1e-10",
"--use_policy_training_preset=false"
```