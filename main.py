import os
import argparse
from pprint import pprint

import torch
import numpy as np

from utils.data import get_data
from utils.model import load_model_and_tokenizer, get_massive_activations_stats
from utils.analysis import (
    check_has_massive_activations, 
    verify_massive_activations_location_and_magnitude_and_tokens, 
    run_intervention_analysis
)
from utils.visualization import plot_top_activation_magnitude, plot_self_attention
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, required=True, help="Type of experiment to run")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--module_name", type=str, default="layer", help="Name of the module in the layer")
    parser.add_argument("--layer_path", type=str, default="model.layers", help="Path to the decoder layers in the model. GPT-2 uses `transformer.h`.")
    parser.add_argument("--attn_path", type=str, default="self_attn", help="Path to the attention layers in the model. GPT-2 uses `attn`.")
    parser.add_argument("--show_logits", type=lambda x: x.lower().startswith("t"), default=True, help="Whether to show logits in the self-attention plot")
    parser.add_argument("--dtype", type=str, default="torch.bfloat16", help="Data type to use")
    parser.add_argument("--add_bos_token", type=lambda x: x.lower().startswith("t"), default=True, help="Whether to add a BOS token")
    parser.add_argument("--context_length", type=int, default=4096, help="Context length")
    parser.add_argument("--sentence", type=str, default="Summer is warm. Winter is cold.", help="Sentence to use for the visualization.")
    parser.add_argument("--save_dir", type=str, default="./massiveactivation_outputs", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Parse arguments
    exp_type = args.exp_type
    pretrained = args.pretrained
    layer_path = args.layer_path
    module_name = args.module_name
    dtype = eval(args.dtype)
    add_bos_token = args.add_bos_token
    context_length = args.context_length
    sentence = args.sentence
    show_logits = args.show_logits
    attn_path = args.attn_path
    save_dir = args.save_dir
    seed = args.seed
    device = args.device

    # Create save directory
    model_name = pretrained.split("/")[-1]
    config_dir = f"{model_name}_{module_name}_{layer_path}_{attn_path}_{context_length}_{add_bos_token}_{show_logits}_{seed}"
    save_dir_sub = os.path.join(save_dir, model_name, config_dir)
    if not os.path.exists(save_dir_sub):
        os.makedirs(save_dir_sub, exist_ok=True)
    with open(os.path.join(save_dir_sub, "config.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Path: {model_name}\n")
        f.write(f"Module Name: {module_name}\n")
        f.write(f"Layer Path: {layer_path}\n")
        f.write(f"Attention Path: {attn_path}\n")
        f.write(f"Context Length: {context_length}\n")
        f.write(f"Add BOS Token: {add_bos_token}\n")
        f.write(f"Show Logits: {show_logits}\n")
        f.write(f"Seed: {seed}\n")

    # Set seed
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(pretrained, 
                                    add_bos_token=add_bos_token,
                                    dtype=dtype,
                                    )

    if exp_type == "existence_validation":
        testseq_list = get_data(
            tokenizer,
            nsamples=10,
            seqlen=context_length,
            add_bos_token=add_bos_token,
            device="cuda",
        )

        has_massive_activations = check_has_massive_activations(
            save_dir_sub, model_name, model, testseq_list, layer_path, module_name, dtype
        )
        print(f"Has massive activations: {has_massive_activations}")
    elif exp_type == "tokens_analysis" or exp_type == "intervention_analysis":
        first_layer, embd_location, tokenswithmassiveact_all_unique = (
            verify_massive_activations_location_and_magnitude_and_tokens(
                save_dir_sub,
                model_name,
                model,
                layer_path,
                tokenizer,
                context_length,
                add_bos_token,
            )
        )
        print(f"First Layer with Massive Activations: {first_layer}")
        print(f"Embedding Position with Massive Activations: {embd_location}")
        print(
            "Unique Tokens (with the Relative Position) with Massive Activations in All Layers:"
        )
        pprint(tokenswithmassiveact_all_unique)

        if exp_type == "intervention_analysis":
            ppl_dict = run_intervention_analysis(
                            save_dir_sub,
                            model_name,
                            model,
                            tokenizer,
                            layer_path,
                            context_length,
                            add_bos_token,
                            seed,
                            first_layer=first_layer,
                            embd_location=embd_location,
                        )
            pprint(ppl_dict)

    elif exp_type == "top_activation_magnitude" or exp_type == "self_attention":
        trajec_seq = "" if not add_bos_token else f"{tokenizer.bos_token}"
        trajec_seq += sentence
        trajec_seq_enc = tokenizer(
            trajec_seq, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        trajec_massive_activations_stats = get_massive_activations_stats(
            trajec_seq_enc, model, tokenizer, layer_path
        )
        if trajec_massive_activations_stats is None:
            trajec_tokenposwithmassiveact = []
        else:
            trajec_tokenposwithmassiveact = trajec_massive_activations_stats[1]

        if exp_type == "self_attention":
            plot_self_attention(
                save_dir_sub,
                model_name,
                model,
                tokenizer,
                layer_path,
                show_logits=show_logits,
                attn_path=attn_path,
                seq=trajec_seq,
                tokenposwithmassiveact=trajec_tokenposwithmassiveact,
            )
        elif exp_type == "top_activation_magnitude":
            plot_top_activation_magnitude(
                save_dir_sub, model_name, model, trajec_seq_enc, layer_path, module_name, dtype
        )