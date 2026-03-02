import torch
from PIL import Image
from app.sana_controlnet_pipeline import SanaControlNetPipeline
from torchvision.utils import save_image
import os
import random
import time

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = SanaControlNetPipeline("./Sana/configs/sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml")
pipe.from_pretrained("/media02/ltnghia31/models/SANA-1.6B-ControlNet/checkpoints/Sana_1600M_1024px_BF16_ControlNet_HED.pth")

# Prepare data
data_path = "./aodai"
sketch_path = f"{data_path}/opensketch_style"
files = os.listdir(sketch_path)
prompts_path = "./prompts_ao_dai_v3.json"
output_path = "./generated_images"
os.makedirs(output_path, exist_ok=True)

with open(prompts_path, 'r', encoding='utf-8') as f:
    import json
    prompt_data = json.load(f)

context_prompts = {
    "wedding / ceremonial attire": (
        "Elegant ceremonial setting with refined traditional Vietnamese décor, "
        "symbolic motifs, soft floral arrangements, and a dignified, formal atmosphere."
    ),

    "Lunar New Year (Tet) celebration": (
        "Festive Tet holiday setting with traditional decorations such as red lanterns, "
        "peach blossoms, kumquat trees, calligraphy banners, and a joyful celebratory mood."
    ),

    "traditional dance performance": (
        "Cultural performance stage with traditional Vietnamese patterns, "
        "flowing fabric banners, theatrical lighting, and a sense of movement and rhythm."
    ),

    "graduation / school event": (
        "Clean academic setting with subtle ceremonial elements, "
        "school architecture, soft daylight, and a formal yet youthful atmosphere."
    ),

    "festival / cultural parade": (
        "Vibrant outdoor festival environment with colorful banners, "
        "crowds in the distance, cultural symbols, and dynamic celebratory energy."
    ),

    "formal portrait fashion": (
        "Minimal high-end portrait backdrop with controlled lighting, "
        "neutral tones, and a refined fashion photography aesthetic."
    ),

    "everyday formal wear": (
        "Subtle lifestyle environment such as a garden walkway or quiet urban space, "
        "natural lighting, and an elegant yet approachable mood."
    ),

    "national heritage demonstration": (
        "Heritage-focused cultural setting highlighting traditional Vietnamese architecture, "
        "historical motifs, and an educational, respectful atmosphere."
    )
}

output_triplet = {}

# Inference
start = time.time()
for ind1 in range(10):
    
    file_name = random.choice(files)
    ref_image = Image.open(f"{sketch_path}/{file_name}")
    name_without_ext = file_name.split(".")[0].split("_")[0]
    # Initialize list for this specific filename if not exists
    if name_without_ext not in output_triplet:
        output_triplet[name_without_ext] = []
    for ind2 in range(10):
        prompt_key = random.choice(list(prompt_data.keys()))
        prompt = prompt_data[prompt_key]
        for ind3 in range(3):
            context_key = random.choice(list(context_prompts.keys()))
            context_desc = context_prompts[context_key]

            full_prompt = f"""
            {prompt}
            Cultural & Contextual Setting:
            {context_desc}
            """

            images = pipe(
                prompt=full_prompt,
                ref_image=ref_image,
                guidance_scale=4.5,
                num_inference_steps=20,
                sketch_thickness=2,
                generator=torch.Generator(device=device).manual_seed(int(time.time()) % 10000),
            )

            save_image(
                images,
                f'./generated_images/{name_without_ext}_{ind2}_{ind3}.png',
                normalize=True,
                value_range=(-1, 1)
            )
            output_triplet[f'{name_without_ext}'].append({
                "sketch": f"{sketch_path}/{file_name}",
                "prompt": prompt_key,
                "context": context_key,
                "full_prompt": full_prompt,
                "generated_image": f'./generated_images/{name_without_ext}_{ind2}_{ind3}.png'
            })

end = time.time()
print(f"Inference time: {end - start} seconds")
with open(f"{output_path}/output_triplet.json", 'w', encoding='utf-8') as f:
    json.dump(output_triplet, f, ensure_ascii=False, indent=4)
# ref_image = Image.open("./aodai/opensketch_style/0000000_out.png")
# prompt = """
# Photorealistic full-length image of a Vietnamese áo dài displayed without a human model, focusing entirely on garment design and construction.
#     Overall impression: Elegant, graceful, youthful, balanced composition, high-end fashion presentation, refined Vietnamese cultural aesthetics, emphasis on silhouette, fabric flow, and craftsmanship, formal portrait fashion.
#     Garment Presentation: The áo dài is shown as a standalone fashion piece (tailored mannequin, invisible form, or suspended display), maintaining natural structure and proportions. The fabric falls realistically with accurate gravity and drape, clearly defining the garment’s shape without visible body parts.
#     Garment Details:
#     Color: black with natural tonal depth and realistic color rendering.
#     Fabric: velvet featuring a soft, flowing drape and detailed textile texture.
#     Silhouette: modern fitted silhouette outlining the traditional áo dài form with smooth, elegant lines. 
#     Floral Embellishments (Pattern): lace-embedded floral style arranged harmoniously across the garment surface, following the contours of the design.
#     Neckline: rounded neckline with a clean, refined finish and precise tailoring.
#     Sleeves: raglan sleeves shown in full length and structure, highlighting stitching and fabric flow.
#     Waist & Closure: traditional loop-and-button subtly integrated into the design.
#     Length & Hem (Skirt): ankle-length tunic with inner pants visible beneath the outer layer, showcasing traditional áo dài layering.
#     Setting & Context:
#     Background: classic architecture scene minimal and unobtrusive, enhancing focus on the garment.
#     Lighting: warm sunset light softly illuminating the fabric, embroidery, and folds, creating gentle shadows and depth."""



# for name, angle_desc in angle_prompts.items():
#     full_prompt = f"""
#     {prompt}
#     Camera & Viewpoint:
#     {angle_desc}
#     """

#     images = pipe(
#         prompt=full_prompt,
#         ref_image=ref_image,
#         guidance_scale=4.5,
#         num_inference_steps=20,
#         sketch_thickness=2,
#         generator=torch.Generator(device=device).manual_seed(42),
#     )

#     save_image(
#         images,
#         f'./generated_images/sana_aodai_{name}.png',
#         normalize=True,
#         value_range=(-1, 1)
#     )