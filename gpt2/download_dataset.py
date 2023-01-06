from datasets import load_dataset

dataset = load_dataset('bigcode/the-stack-dedup',
                       use_auth_token=True,
                       split='train',
                       cache_dir='stack_dedup_python',
                       data_dir='data/python')

dataset.to_json('stack_python.json', num_proc=16)
