from datasets import load_dataset
# Load Common Crawl dataset with streaming
dataset = load_dataset("allenai/c4", "en", split="train",
                       streaming=True,trust_remote_code=True)
print(next(iter(dataset)))

with open("sample_data.txt", "w") as f:
    for i,sample in enumerate(dataset):
        f.write(sample["text"] + "\n")
        if i >= 1000:
            break
print(f"Downloaded {i} samples and saved to 'sample_data.txt'")