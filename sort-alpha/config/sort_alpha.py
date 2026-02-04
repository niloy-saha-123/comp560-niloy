# Train a model to sort characters alphabetically

out_dir = 'out/sort-alpha'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 

# always save
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'comp560-sort'
wandb_run_name = 'sort-alpha-basic'

dataset = 'sort-alpha' # We will look for data in data/sort-alpha relative to this repo root? 


dataset = 'sort' 

gradient_accumulation_steps = 1
batch_size = 64
block_size = 64 # strings are short (3-5 chars * 2 + 2 = ~12 chars max). 64 is plenty.

# Small GPT model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000 
lr_decay_iters = 2000 
min_lr = 1e-4 
beta2 = 0.99 

warmup_iters = 100 

device = 'mps'  
compile = False 
