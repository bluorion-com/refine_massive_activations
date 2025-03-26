import math
import os
from copy import deepcopy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from monkey_patch.custom_attn import get_custom_attn_func
from utils.model import (
    add_global_plot_styles, 
    check_module_hidden_states, 
    get_layer_outputs_with_custom_module,
)


def plot_top_activation_magnitude(
    save_dir_sub, model_name, model, seq, layer_path, module_name, dtype
):
    magnitude, index, hidden_states_magnitude, hidden_states_median = check_module_hidden_states(
        model, [seq], layer_path, module_name, input_or_output="output", dtype=dtype
    )
    fig = plt.figure(figsize=(8, 5))
    add_global_plot_styles()
    plt.plot(
        range(len(magnitude)),
        hidden_states_magnitude,
        color="blue",
        marker="o",
        markersize=5,
        label="top-1 hidden states",
    )
    plt.plot(
        range(len(magnitude)),
        hidden_states_median,
        color="gray",
        marker="o",
        markersize=5,
        label="median hidden states",
    )
    plt.xlabel("Layer Number")
    plt.ylabel("Activation Magnitude")
    plt.title(model_name)
    plt.yticks(rotation=90, va="center")
    plt.legend()
    plt.savefig(
        os.path.join(save_dir_sub, "top_activation_magnitude.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)



def plot_self_attention(
    save_dir_sub,
    model_name,
    model,
    tokenizer,
    layer_path,
    show_logits=True,
    attn_path="self_attn",
    seq="The following are multiple choice questions (with answers) about machine learning.\n\n A 6-sided die is rolled 15 times and the results are: side 1 comes up 0 times;",
    tokenposwithmassiveact=None,
):
    valenc = tokenizer(seq, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    seq_decoded = []
    for i in range(valenc.shape[1]):
        seq_decoded.append(tokenizer.decode(valenc[0, i].item()))

    if "kv_bias" in model_name.lower() or "kvbias" in model_name.lower():
        x_seq_decoded = deepcopy(seq_decoded)
        x_seq_decoded.insert(0, "extra_kv")

        if tokenposwithmassiveact:
            tokenposwithmassiveact = [0] + [x + 1 for x in tokenposwithmassiveact]
    else:
        x_seq_decoded = seq_decoded

    custom_attn_func = get_custom_attn_func(model_name)
    layer_outputs, original_forward_dict = get_layer_outputs_with_custom_module(
        valenc, model, layer_path, custom_attn_func, attn_path
    )

    num_layers = len(layer_outputs)
    cols = 4
    rows = math.ceil(num_layers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.tight_layout()  # Adjust layout
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust spacing between subplots

    for layer_id, (layer, output) in enumerate(layer_outputs.items()):
        ax = axes[layer_id // cols, layer_id % cols]
        if attn_path is None:
            attn_layer = layer
        else:
            attn_layer = getattr(layer, attn_path)
        if show_logits:  # before softmax
            attn = attn_layer.attn_logits.detach().cpu()
        else:  # after softmax
            attn = attn_layer.attn_probs.detach().cpu()

        attn_layer.forward = original_forward_dict[layer_id]

        corr = attn.float().numpy()[0].mean(0)  # avg over heads
        corr = corr.astype("float64")
        plot_attn_sub(
            ax,
            corr,
            layer_id,
            x_seq_decoded,
            seq_decoded,
            tokenposwithmassiveact,
            is_last_row=layer_id // cols == rows - 1,
            is_first_col=layer_id % cols == 0,
        )

    # Hide any unused subplots if the grid is larger than the number of layers
    for idx in range(len(layer_outputs), rows * cols):
        fig.delaxes(axes.flatten()[idx])

    plt.savefig(
        os.path.join(save_dir_sub, "self_attention.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

def plot_attn_sub(
    ax,
    corr,
    layer_id,
    seqlist_x=[],
    seqlist_y=[],
    special_seqlist=[],
    is_last_row=False,
    is_first_col=False,
):
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(
        corr,
        mask=mask,
        square=True,
        ax=ax,
        cmap="YlGnBu",
        cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect": 50},
    )

    ax.set_facecolor("whitesmoke")
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=18)

    ax.tick_params(axis="x", which="major")
    ax.set_xticklabels(seqlist_x, rotation=90)
    ax.set_yticklabels(seqlist_y, rotation=0)
    if is_last_row:
        ax.set_xlabel("Key", fontsize=18, fontweight="bold")
    if is_first_col:
        ax.set_ylabel("Query", fontsize=18, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    ax.set_title(f"Layer {layer_id + 1}", fontsize=24, fontweight="bold")

    for x_id in special_seqlist:
        ax.get_xticklabels()[x_id].set_weight("heavy")
        ax.get_xticklabels()[x_id].set_color("red")