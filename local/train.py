from datasets import load_dataset

ds = load_dataset("ILSVRC/imagenet-1k")

print(ds)

print(ds["train"][0])