from __future__ import annotations
import argparse, torch
from inference.inference import load_trained_model  


from collator_sft import SFTCollator
from formatters import format_prompt_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--tokens', type=int, default=80)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default='tokenizer')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})

    col = SFTCollator(block_size=cfg.get('block_size', 256), bpe_dir=args.bpe_dir)
    model,config = load_trained_model("models/final_model.pt")
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    prompt_text = format_prompt_only(args.prompt).replace('</s>','')
    ids = col.encode(prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.tokens, 
                             temperature=args.temperature, top_k=3)

    # decode: prefer BPE if collator has it, else fall back to bytes
    out_ids = out[0].tolist()
    orig_len = idx.size(1)
    if hasattr(col, "tok") and hasattr(col.tok, "decode"):
        # decode full text or just the generated suffix; suffix is often clearer
        generated = col.tok.decode(out_ids)
        print(generated)
    else:
        generated = bytes(out_ids[orig_len:]).decode("utf-8", errors="ignore")
        print(generated)


if __name__ == '__main__':
    main()