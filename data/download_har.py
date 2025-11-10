from datasets import load_dataset
import pandas as pd

ds = load_dataset("Bingsu/Human_Action_Recognition")
print("Dataset Loaded")

print(f"ds[train] is {ds["train"][0]["image"]}")