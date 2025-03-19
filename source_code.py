import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
from IPython.display import display

# Load the LLaMA model for prompt expansion
def expand_prompt(user_input):
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change to your fine-tuned model if needed

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Format prompt for LLaMA
    input_text = f"Make this image generation prompt more creative and detailed: {user_input}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.8, top_p=0.95)

    # Decode and return refined prompt
    refined_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
    return refined_prompt

# Generate an image from text using Stable Diffusion
def generate_image_from_text(prompt, output_path="generated_image.png"):
    model_id = "runwayml/stable-diffusion-v1-5"

    # Load Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )

    # Move to the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    # Generate image
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image saved as {output_path}")

    # Display in Colab
    display(image)

    return image

# Main execution
if __name__ == "__main__":
    user_input = "a futuristic city"  # Example user input
    refined_prompt = expand_prompt(user_input)
    
    print(f"Refined Prompt: {refined_prompt}")  # Debugging output
    
    generated_image = generate_image_from_text(refined_prompt)

