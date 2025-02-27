
import torch
from transformers import pipeline, SiglipModel, AutoProcessor
import numpy as np
import gradio as gr

cache_dir = 'volumes'

siglip_checkpoint = "nielsr/siglip-base-patch16-224"
clip_checkpoint = "openai/clip-vit-base-patch16"
clip_detector = pipeline(model=clip_checkpoint, task="zero-shot-image-classification", cache_dir=cache_dir)
siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224", cache_dir=cache_dir)
siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", cache_dir=cache_dir)


def postprocess(output):
  return {out["label"]: float(out["score"]) for out in output}

def postprocess_siglip(output, labels):
  return {labels[i]: float(np.array(output[0])[i]) for i in range(len(labels))}
    
def siglip_detector(image, texts):
  inputs = siglip_processor(text=texts, images=image, return_tensors="pt",
                     padding="max_length")

  with torch.no_grad():
    outputs = siglip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image) 
  return probs


def infer(image, candidate_labels):
  candidate_labels = [label.lstrip(" ") for label in candidate_labels.split(",")]
  siglip_out = siglip_detector(image, candidate_labels)
  clip_out = clip_detector(image, candidate_labels=candidate_labels)
  return postprocess(clip_out), postprocess_siglip(siglip_out, labels=candidate_labels)


with gr.Blocks() as demo:
  gr.Markdown("# Compare CLIP and SigLIP")
  gr.Markdown("Compare the performance of CLIP and SigLIP on zero-shot classification in this Space 👇")
  with gr.Row():
    with gr.Column():
        image_input = gr.Image(type="pil")
        text_input = gr.Textbox(label="Input a list of labels")
        run_button = gr.Button("Run", visible=True)

    with gr.Column():
      clip_output = gr.Label(label = "CLIP Output", num_top_classes=3)
      siglip_output = gr.Label(label = "SigLIP Output", num_top_classes=3)
      
  examples = [["C:\\Users\\Dell\\Downloads\\test_image.jpg", "Cubism, Surrealism, Impressionism, Abstract Expressionism, Realism, Minimalism, Art Nouveau, Fauvism, Dadaism, Baroque"]]

  gr.Examples(
        examples = examples, 
        inputs=[image_input, text_input],
        outputs=[clip_output, 
                 siglip_output
                 ],
        fn=infer,
        cache_examples=True
    )
  run_button.click(fn=infer,
                    inputs=[image_input, text_input],
                    outputs=[clip_output, 
                             siglip_output
                             ])

demo.launch()
