# Configuration for training small models for interpretability analysis
# Optimized for quick training and easy analysis

# I/O
out_dir = 'out-interpretability'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'nanogpt-interpretability'
wandb_run_name = 'small-interp-model'

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128  # Shorter context for faster training

# model - Small but capable of learning induction
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# adamw optimizer
learning_rate = 1e-3
max_iters = 5000  # Quick training
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
