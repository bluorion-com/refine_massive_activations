import types
import matplotlib.pyplot as plt

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MASSIVE_ACTIVATION_ABS_THRESHOLD = 100
MASSIVE_ACTIVATION_REL_THRESHOLD = 1000


def load_model_and_tokenizer(pretrained, add_bos_token, dtype = torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
    model = model.to("cuda")

    if add_bos_token:
        if tokenizer.bos_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.bos_token = tokenizer.eos_token
                print(
                    f"BOS token is not available. Setting BOS token to EOS token: {tokenizer.eos_token}"
                )
            else:
                raise ValueError("BOS and EOS token is not available.")
    return model, tokenizer

def add_global_plot_styles(multiplier: int = 1):
    plt.rcParams["font.family"] = "serif"
    plt.grid(color="#CCCCCC", linestyle="--")
    plt.rcParams.update(
        {
            "axes.titlesize": 17 * multiplier,
            "axes.labelsize": 15 * multiplier,
            "xtick.labelsize": 10 * multiplier,
            "ytick.labelsize": 10 * multiplier,
        }
    )
    plt.tight_layout()

def check_module_hidden_states(
    model,
    inputs_list,
    layer_path,
    module_name,
    input_or_output="output",
    dtype=torch.float16,
):
    if input_or_output not in ["input", "output"]:
        raise ValueError(
            "input_or_output should be 'input' or 'output', instead of", input_or_output
        )

    if dtype == torch.bfloat16:
        dtype = torch.float16

    all_activations = {}
    layer_outputs = {}
    all_hooks = []

    def get_activations(layer_index):
        def hook(model, inputs, outputs):
            hidden_states = inputs if input_or_output == "input" else outputs
            all_activations.setdefault(layer_index, {})[
                f"{module_name}_{input_or_output}_hidden_states"
            ] = hidden_states

        return hook

    def get_hidden_states(module, input, output):
        layer_outputs[module] = output

    attributes = module_name.split(".") if module_name != "layer" else []
    layers = get_layers(model, layer_path)

    for layer_index, layer in enumerate(layers):
        current_attr = layer
        valid = True
        for attr in attributes:
            if hasattr(current_attr, attr):
                current_attr = getattr(current_attr, attr)
            else:
                valid = False
                break

        if valid:
            hook = current_attr.register_forward_hook(get_activations(layer_index))
            all_hooks.append(hook)

        hook = layer.register_forward_hook(get_hidden_states)
        all_hooks.append(hook)

    model.eval()
    top1_values_all_layers = []
    top1_indexes_all_layers = []
    top1_hidden_states_all_layers = []
    median_hidden_states_all_layers = []
    for inputs in inputs_list:
        with torch.no_grad():
            model(inputs)

        top1_values_all_layers_i = []
        top1_indexes_all_layers_i = []
        for layer_index, outputs in all_activations.items():
            values = outputs[f"{module_name}_{input_or_output}_hidden_states"]
            tensor = values[0] if isinstance(values, tuple) else values
            tensor = tensor.detach().cpu()
            tensor_abs = tensor.view(-1).abs().float()

            max_value, max_index = torch.max(tensor_abs, 0)
            max_index = torch.unravel_index(max_index, tensor.shape)
            top1_values_all_layers_i.append(max_value)
            top1_indexes_all_layers_i.append(tuple(max_index))

        top1_values_all_layers.append(top1_values_all_layers_i)
        top1_indexes_all_layers.append(top1_indexes_all_layers_i)

        top1_hidden_states_all_layers_i = []
        median_hidden_states_all_layers_i = []
        for layer, output in layer_outputs.items():
            tensor = output[0].detach().cpu()
            tensor_abs = tensor.view(-1).abs().float()

            max_value, max_index = torch.max(tensor_abs, 0)

            top1_hidden_states_all_layers_i.append(max_value)
            median_hidden_states_all_layers_i.append(torch.median(tensor_abs))

        top1_hidden_states_all_layers.append(top1_hidden_states_all_layers_i)
        median_hidden_states_all_layers.append(median_hidden_states_all_layers_i)

    for hook in all_hooks:
        hook.remove()

    # Convert to Tensor and Array
    top1_values_all_layers = torch.Tensor(top1_values_all_layers)
    top1_indexes_all_layers = np.array(top1_indexes_all_layers)
    top1_hidden_states_all_layers = torch.Tensor(top1_hidden_states_all_layers)
    median_hidden_states_all_layers = torch.Tensor(median_hidden_states_all_layers)

    # Aggregate across samples
    top1_values_all_layers_max_out = torch.max(top1_values_all_layers, dim=0)
    top1_values_all_layers = top1_values_all_layers_max_out.values.to(dtype)
    top1_values_all_layers_indices = top1_values_all_layers_max_out.indices
    top1_indexes_all_layers = [
        tuple(x[:, i])
        for i, x in zip(top1_values_all_layers_indices, top1_indexes_all_layers.transpose(1, 2, 0))
    ]
    top1_hidden_states_all_layers = torch.mean(top1_hidden_states_all_layers, dim=0).to(dtype)
    median_hidden_states_all_layers = torch.mean(median_hidden_states_all_layers, dim=0).to(dtype)

    return (
        top1_values_all_layers,
        top1_indexes_all_layers,
        top1_hidden_states_all_layers,
        median_hidden_states_all_layers,
    )

def get_layers(model, layer_path):
    attributes = layer_path.split(".")
    layers = model
    for attr in attributes:
        layers = getattr(layers, attr)
    return layers

def get_layer_outputs_with_custom_module(
    inputs, model, layer_path, custom_module_func, module_path="mlp"
):
    layer_outputs = {}
    all_hooks = []
    original_forward = {}

    def get_hidden_states(module, input, output):
        layer_outputs[module] = output

    layers = get_layers(model, layer_path)
    for layer_index, layer in enumerate(layers):
        if module_path is None:
            ori_layer = layer
            original_forward[layer_index] = ori_layer.forward
            modified_layer = ori_layer
            # Replace forward function with custom forward
            modified_layer.forward = types.MethodType(custom_module_func, modified_layer)
        else:
            ori_module = getattr(layer, module_path)
            original_forward[layer_index] = ori_module.forward
            modified_module = ori_module
            # Replace mlp layer with custom mlp
            modified_module.forward = types.MethodType(custom_module_func, modified_module)

        hook = layer.register_forward_hook(get_hidden_states)
        all_hooks.append(hook)

    with torch.no_grad():
        model(inputs)

    for hook in all_hooks:
        hook.remove()

    return layer_outputs, original_forward

def get_layer_outputs(inputs, model, layer_path):
    layer_outputs = {}
    all_hooks = []

    def get_hidden_states(module, input, output):
        layer_outputs[module] = output

    layers = get_layers(model, layer_path)
    for layer_index, layer in enumerate(layers):
        hook = layer.register_forward_hook(get_hidden_states)
        all_hooks.append(hook)

    with torch.no_grad():
        model(inputs)

    for hook in all_hooks:
        hook.remove()

    return layer_outputs

def get_massive_activations_stats(valenc, model, tokenizer, layer_path):
    layer_outputs = get_layer_outputs(valenc, model, layer_path)
    seq_decoded = []
    for i in range(valenc.shape[1]):
        seq_decoded.append(tokenizer.decode(valenc[0, i].item()))
    massive_activations_per_layer = get_massive_activations_per_layer(
        seq_decoded, layer_outputs, silent=True
    )
    layers_with_massive_act = [k for k, v in massive_activations_per_layer.items() if v[0]]
    if not layers_with_massive_act:
        return None

    first_layer_pos_with_massive_act = sorted(layers_with_massive_act)[0]

    tokenposwithmassiveact, massiveactposinembd, massiveactmaginembd, stats, _ = (
        massive_activations_per_layer[first_layer_pos_with_massive_act]
    )
    return (
        first_layer_pos_with_massive_act,
        tokenposwithmassiveact,
        massiveactposinembd,
        massiveactmaginembd,
        stats,
        massive_activations_per_layer,
    )

def get_massive_activations_per_layer(seq_decoded, layer_outputs, silent=False, write_file=None):
    massive_activations_per_layer = {}
    for layer_id, (layer, output) in enumerate(layer_outputs.items()):
        stats = {}
        stats["seq"] = seq_decoded
        tensor_abs = output[0].detach().cpu().abs().float()
        stats[f"{layer_id}"] = tensor_abs
        stats[f"{layer_id}_ori"] = output[0].detach().cpu().float()

        max_abs = float(torch.max(tensor_abs).item())
        median_abs = float(torch.median(tensor_abs).item())
        if not silent:
            if write_file is not None:
                with open(write_file, "a") as f:
                    f.write(f"Layer-{layer_id} | Max Activation: {max_abs}\n")
            print(f"Layer-{layer_id} | Max Activation: {max_abs}")
        if (
            max_abs > MASSIVE_ACTIVATION_ABS_THRESHOLD
            and max_abs > MASSIVE_ACTIVATION_REL_THRESHOLD * median_abs
        ):

            tokenposwithmassiveact = []
            massiveactposinembd = []
            massiveactmaginembd = {}
            for tokenpos, embd in enumerate(tensor_abs[0]):
                max_embd = float(torch.max(embd).item())
                # median_embd = float(torch.median(embd).item())
                if (
                    max_embd > MASSIVE_ACTIVATION_ABS_THRESHOLD
                    and max_embd > MASSIVE_ACTIVATION_REL_THRESHOLD * median_abs
                ):
                    tokenposwithmassiveact.append(tokenpos)
                    massiveactpos = (
                        torch.where(
                            (embd > MASSIVE_ACTIVATION_ABS_THRESHOLD)
                            & (embd > MASSIVE_ACTIVATION_REL_THRESHOLD * median_abs)
                        )[0]
                        .numpy()
                        .tolist()
                    )

                    massiveactmaginembd[tokenpos] = {}
                    for pos in massiveactpos:
                        if pos not in massiveactposinembd:
                            massiveactposinembd.append(pos)
                        massiveactmaginembd[tokenpos][pos] = float(embd[pos].item())
            if not silent:
                if write_file is not None:
                    with open(write_file, "a") as f:
                        f.write(
                            f"Token Positions with Massive Activations: {tokenposwithmassiveact}\n"
                        )
                        f.write(
                            f"Massive Activation Positions in Embedding: {massiveactposinembd}\n"
                        )
                print(f"Token Positions with Massive Activations: {tokenposwithmassiveact}")
                print(f"Massive Activation Positions in Embedding: {massiveactposinembd}")

            massiveactposinembd = sorted(massiveactposinembd)
        else:
            tokenposwithmassiveact = []
            massiveactposinembd = []
            massiveactmaginembd = {}
            if not silent:
                if write_file is not None:
                    with open(write_file, "a") as f:
                        f.write("No massive activations found in this layer.\n")
                print("No massive activations found in this layer.")

        massive_activations_per_layer[layer_id] = (
            tokenposwithmassiveact,
            massiveactposinembd,
            massiveactmaginembd,
            stats,
            max_abs,
        )
    return massive_activations_per_layer

def reset_first_massive_activations(
    model,
    layer_path,
    first_layer_pos_with_massive_act,
    massiveactposinembd_resetvalue_dict,
    is_starting=True,
    token_dim=None,
    write_file=None,
):
    def set_to_zero(
        module, input, output, token_dim, massiveactposinembd_resetvalue_dict, is_starting
    ):
        out_feature = output[0]

        if token_dim is None:
            if not is_starting:
                feat_abs = out_feature.abs()
                sort_res = torch.sort(feat_abs.flatten(), descending=True)
                top_indices = sort_res.indices[0]
                token_dim = top_indices.item() // feat_abs.shape[2]
            else:
                token_dim = 0
        for loc, reset_value in massiveactposinembd_resetvalue_dict.items():
            orig_value = out_feature[:, token_dim, loc].cpu()
            if write_file is not None:
                with open(write_file, "a") as f:
                    f.write(
                        f"Resetting massive activations to {reset_value} in layer {layer_index} token_dim {token_dim} loc {loc}. Original Value: {orig_value}\n"
                    )
            out_feature[:, token_dim, loc] = reset_value
        return (out_feature, *output[1:])

    layers = get_layers(model, layer_path)
    for layer_index, layer in enumerate(layers):
        if layer_index == first_layer_pos_with_massive_act:
            # print(f"Resetting massive activations to {reset_value} in layer {layer_index}")
            hook = layer.register_forward_hook(
                lambda module, input, output: set_to_zero(
                    module,
                    input,
                    output,
                    token_dim,
                    massiveactposinembd_resetvalue_dict,
                    is_starting,
                )
            )
            break

    return hook