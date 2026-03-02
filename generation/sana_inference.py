import torch
from PIL import Image
from app.sana_controlnet_pipeline import SanaControlNetPipeline
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = SanaControlNetPipeline("./Sana/configs/sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml")
# pipe.vae.to(device=pipe.device, dtype=torch.float16) #=> Fix for type casting between input and auto encoder.
pipe.from_pretrained("/media02/ltnghia31/models/SANA-1.6B-ControlNet/checkpoints/Sana_1600M_1024px_BF16_ControlNet_HED.pth")

# pipe.vae.to(torch.float16)
# pipe.text_encoder.to(torch.float16)

ref_image = Image.open("./aodai/opensketch_style/0000000_out.png")
prompt = """
Photorealistic full-length image of a Vietnamese áo dài displayed without a human model, focusing entirely on garment design and construction, hanging naturally on a wooden hanger.

    Overall impression: Elegant, graceful, youthful, balanced composition, high-end fashion presentation, refined Vietnamese cultural aesthetics, emphasis on silhouette, fabric flow, and craftsmanship, formal portrait fashion.
    Garment Presentation: The áo dài is shown as a standalone fashion piece (tailored mannequin, invisible form, or suspended display), maintaining natural structure and proportions. The fabric falls realistically with accurate gravity and drape, clearly defining the garment’s shape without visible body parts.
    Garment Details:
    Color: black with natural tonal depth and realistic color rendering.
    Fabric: velvet featuring a soft, flowing drape and detailed textile texture.
    Silhouette: modern fitted silhouette outlining the traditional áo dài form with smooth, elegant lines. 
    Floral Embellishments (Pattern): lace-embedded floral style arranged harmoniously across the garment surface, following the contours of the design.
    Neckline: rounded neckline with a clean, refined finish and precise tailoring.
    Sleeves: raglan sleeves shown in full length and structure, highlighting stitching and fabric flow.
    Waist & Closure: traditional loop-and-button subtly integrated into the design.
    Length & Hem (Skirt): ankle-length tunic with inner pants visible beneath the outer layer, showcasing traditional áo dài layering.
    Display & Physics:
    The áo dài is suspended from a wooden hanger attached to a horizontal rack.
    Setting & Context:
    Background: classic architecture scene minimal and unobtrusive, enhancing focus on the garment.
    Lighting: warm sunset light softly illuminating the fabric, embroidery, and folds, creating gentle shadows and depth."""
# ref_image = Image.open("./Sana/asset/controlnet/ref_images/A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a la.jpg")
# prompt = "A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a landscape."
# with torch.no_grad():
#     with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
# with torch.cuda.amp.autocast(dtype=torch.float32):
images = pipe(
    prompt=prompt,
    ref_image=ref_image,
    guidance_scale=4.5,
    num_inference_steps=20,
    sketch_thickness=2,
    generator=torch.Generator(device=device).manual_seed(0),
)

save_image(images, './generated_images/sana_so2.png', normalize=True, value_range=(-1, 1))