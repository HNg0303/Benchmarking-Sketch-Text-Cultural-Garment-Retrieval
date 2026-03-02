import json
import random

# Load the triplet.json file
with open('aodai/captions/triplet.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. Group every 3 adjacent entries into triplets FIRST (to keep relationships)
triplets = []
for i in range(0, len(data), 3):
    # Safety check to ensure we have a full set of 3
    if i + 2 < len(data):
        triplet = {
            "sketch": data[i]["sketch"],
            "caption": data[i]["caption"],
            "images": [data[i]["image"], data[i+1]["image"], data[i+2]["image"]],
            "original_entries": [data[i], data[i+1], data[i+2]] # Temporary storage for flattening
        }
        triplets.append(triplet)

# 2. Shuffle the triplets
random.shuffle(triplets)

# 3. Split the data
total = len(triplets)
train_size = int(0.8 * total)
val_size = int(0.1 * total)

train_triplets = triplets[:train_size]
val_data = triplets[train_size:train_size + val_size]
test_data = triplets[train_size + val_size:]

# 4. Flatten the training data back to the original type
# We extract the individual items from our "original_entries" bucket
flattened_train = []
for t in train_triplets:
    flattened_train.extend(t["original_entries"])

# 5. Clean up val/test (remove the temporary storage used for flattening)
for d in val_data + test_data:
    if "original_entries" in d:
        del d["original_entries"]

# Save to JSON files
with open('./aodai/captions/cap.train.json', 'w', encoding='utf-8') as f:
    json.dump(flattened_train, f, ensure_ascii=False, indent=4)

with open('./aodai/captions/cap.val.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open('./aodai/captions/cap.test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"Processed {total} total triplets.")
print(f"Train (Flattened): {len(flattened_train)} individual items")
print(f"Val (Grouped): {len(val_data)} triplets")
print(f"Test (Grouped): {len(test_data)} triplets")