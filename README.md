## Rethink Agentic World Models

### WebShop RL Training

```bash
cd /fs_mol/liujiarun/rethink-agentic-wm
bash scripts/webshop/build_search_index.sh
```

```bash
python scripts/webshop/prepare_webshop_data.py \
    --webshop-path /fs_mol/liujiarun/WebShop \
    --output-dir   /fs_mol/liujiarun/rethink-agentic-wm/data/webshop \
    --train-range  1500 6910 \
    --val-range    500 1500
```

```bash
cd /fs_mol/liujiarun/rethink-agentic-wm
bash scripts/webshop/start_env_server.sh
```

```bash
conda activate verl-0.9.1
cd /fs_mol/liujiarun/rethink-agentic-wm
bash scripts/webshop/train_grpo_qwen25.sh
```


