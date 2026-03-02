import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"JSON saved to {file_path}")


if __name__ == "__main__":
    captions_json = "./generated/outputs_ao_dai_caption_refined.json"
    prompts_json = "./generated/output_triplet.json"
    triplet_json = "./generated/output_triplet_final.json"

    triplet = load_json(triplet_json)
    refined_triplet = []
    split_sketches = []
    split_images = []
    for item in triplet:
        new_item = {}
        new_item["sketch"] = item["sketch"].split("/")[-1]
        new_item["caption"] = item["caption"]
        new_item["image"] = item["image"].split("/")[-1]
        split_sketches.append(item["sketch"].split("/")[-1])
        split_images.append(item["image"].split("/")[-1])
        refined_triplet.append(new_item)
    
    save_json(refined_triplet, "./triplet.json")
    print(f"Refined JSON saved to ./triplet.json")
    save_json(split_sketches, "./split.sketches.json")
    print(f"Refined JSON saved to ./split.sketches.json")
    save_json(split_images, "./split.images.json")
    print(f"Refined JSON saved to ./split.images.json")

