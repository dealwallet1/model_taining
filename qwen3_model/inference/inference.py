import torch
import torch.nn as nn  
import torch.nn.functional as F  

from transformers import AutoTokenizer 

from configuration.model_config import ModelConfig
from core.llm import MinimalLLM

#need to change model path
def load_trained_model(model_path: str = "final_model.pt"):
    """Load a trained model from checkpoint"""
    # Add ModelConfig to safe globals for PyTorch 2.6+
    from torch.serialization import add_safe_globals
    add_safe_globals([ModelConfig])

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
    except Exception as e:
        print(f"⚠️ Error loading with weights_only=True, trying with weights_only=False...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

    # Create model with same config
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")
    return model, config

@torch.no_grad()
def generate_text(model: nn.Module, tokenizer, prompt: str, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
    """Generate text using the trained model"""
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits, _ = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence - FIX: ensure same dimensions
            next_token = next_token.unsqueeze(0)  # Add batch dimension
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if we reach the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

@torch.no_grad()
def interactive_inference(model_path: str = "final_model.pt"):
    """Interactive inference session"""
    print("🤖 Starting interactive inference session")
    print("Type 'quit' to exit")

    # Load model and tokenizer
    model, config = load_trained_model(model_path)

    # Load tokenizer (assuming we have the same one used during training)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    while True:
        try:
            prompt = input("\n Enter your prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not prompt.strip():
                continue

            print("🔄 Generating...")
            generated_text = generate_text(
                model, tokenizer, prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )

            print(f"\n Generated text:")
            print(f"📝 {generated_text}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

@torch.no_grad()
def demo_inference(model_path: str = "final_model.pt"):
    """Run a quick demo of the model's capabilities"""
    print("🎭 Running inference demo")

    # Load model and tokenizer
    model, config = load_trained_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Demo prompts
    demo_prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important thing to remember is",
        "In the year 2050, technology will",
        "The best way to learn programming is"
    ]

    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n Demo {i}: '{prompt}'")
        print("-" * 50)

        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=100,
            temperature=0.7,
            top_k=40,
            top_p=0.85
        )

        print(f"📝 {generated_text}")
        print()
    
@torch.no_grad()
def start_inference():
    # Check if we have a trained model
    import os

    if os.path.exists("final_model.pt"):
        print("🎉 Found trained model! Running demo...")
        demo_inference("final_model.pt")

        # Optionally run interactive session
        response = input("\n🤖 Would you like to try interactive inference? (y/n): ")
        if response.lower() in ['y', 'yes']:
            interactive_inference("output/models/final_model.pt")
    else:
        print("⚠️ No trained model found. Please run the training cells first.")
        print("💡 Look for 'output/models/final_model.pt' or 'output/models/best_model.pt' in your directory.")