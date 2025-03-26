import os
from tqdm import tqdm
import shutil

import torch
import numpy as np

from utils.data import get_data, get_test_data
from utils.model import (
    check_module_hidden_states, 
    get_massive_activations_stats,
    reset_first_massive_activations,
    MASSIVE_ACTIVATION_ABS_THRESHOLD, 
    MASSIVE_ACTIVATION_REL_THRESHOLD
)

def check_has_massive_activations(
    save_dir_sub, model_name, model, testseq_list, layer_path, module_name, dtype
):

    _, _, hidden_states_magnitude, hidden_states_median = check_module_hidden_states(
        model, testseq_list, layer_path, module_name, input_or_output="output", dtype=dtype
    )
    if os.path.exists(os.path.join(save_dir_sub, "check_has_massive_activations.txt")):
        os.remove(os.path.join(save_dir_sub, "check_has_massive_activations.txt"))
    with open(os.path.join(save_dir_sub, "check_has_massive_activations.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("Location & Value of Massive Activations\n")
        f.write("=" * 50 + "\n")
        has_massive_activations = False
        for layer, values in enumerate(hidden_states_magnitude):
            if (
                values > MASSIVE_ACTIVATION_ABS_THRESHOLD
                and values >= MASSIVE_ACTIVATION_REL_THRESHOLD * hidden_states_median[layer]
            ):
                f.write(f"Layer-{layer}: {values}\n")
                has_massive_activations = True

    return has_massive_activations

def verify_massive_activations_location_and_magnitude_and_tokens(
    save_dir_sub,
    model,
    layer_path,
    tokenizer,
    context_length,
    add_bos_token,
):
    calibrationdata = get_data(
        tokenizer,
        nsamples=50,
        seqlen=context_length,
        add_bos_token=add_bos_token,
        device="cuda",
    )

    first_layer = float("inf")
    embd_location_firstlayer = set()
    massiveactmagnitude_firstlayer = {}
    tokenswithmassiveact_all_unique = {}
    if os.path.exists(
        os.path.join(save_dir_sub, "verify_massive_activations_location_and_magnitude_and_tokens.txt")
    ):
        os.remove(
            os.path.join(save_dir_sub, "verify_massive_activations_location_and_magnitude_and_tokens.txt")
        )
    with open(
        os.path.join(save_dir_sub, "verify_massive_activations_location_and_magnitude_and_tokens.txt"), "w"
    ) as f:
        data_i = 0
        for valenc in tqdm(calibrationdata):
            massive_activation_stats = get_massive_activations_stats(
                valenc, model, tokenizer, layer_path
            )
            if massive_activation_stats is None:
                continue
            else:
                (
                    first_layer_pos_with_massive_act,
                    _,
                    massiveactposinembd,
                    _,
                    _,
                    massive_activations_per_layer,
                ) = massive_activation_stats
                layers_with_massive_act = [
                    k for k, v in massive_activations_per_layer.items() if v[0]
                ]
            if first_layer_pos_with_massive_act < first_layer:
                first_layer = first_layer_pos_with_massive_act
            if len(embd_location_firstlayer) == 0:
                embd_location_firstlayer = set(massiveactposinembd)
            else:
                # get all embd locations with massive activations from all layers
                # and union
                for layer in layers_with_massive_act:
                    (
                        _,
                        massiveactposinembd_perlayer,
                        _,
                        _,
                        _,
                    ) = massive_activations_per_layer[layer]
                    embd_location_firstlayer = embd_location_firstlayer.union(
                        set(massiveactposinembd_perlayer)
                    )

            f.write(f"Datum {data_i}\n")
            f.write(f"Layers with Massive Activations: {layers_with_massive_act}\n")
            for layer in layers_with_massive_act:
                (
                    tokenposwithmassiveact_perlayer,
                    massiveactposinembd_perlayer,
                    massiveactmaginembd_perlayer,
                    _,
                    _,
                ) = massive_activations_per_layer[layer]
                tokenswithmassiveact = {
                    i: tokenizer.decode(valenc[0, i].item())
                    for i in tokenposwithmassiveact_perlayer
                }
                for k, v in tokenswithmassiveact.items():
                    if k not in tokenswithmassiveact_all_unique:
                        tokenswithmassiveact_all_unique[k] = set()
                    tokenswithmassiveact_all_unique[k].add(v)
                if layer == first_layer:
                    for tokenpos, pos_dict in massiveactmaginembd_perlayer.items():
                        if tokenpos not in massiveactmagnitude_firstlayer:
                            massiveactmagnitude_firstlayer[tokenpos] = {}
                        for pos, mag in pos_dict.items():
                            if pos not in massiveactmagnitude_firstlayer[tokenpos]:
                                massiveactmagnitude_firstlayer[tokenpos][pos] = []

                            massiveactmagnitude_firstlayer[tokenpos][pos].append(mag)

                f.write(f"Layer {layer}\n")
                f.write(
                    f"Token Positions with Massive Activations: {tokenposwithmassiveact_perlayer}\n"
                )
                f.write(f"Associated Tokens with Massive Activations: {tokenswithmassiveact}\n")
                f.write("Embedding Position with Massive Activations:\n")
                f.write(f"----- Position: {massiveactposinembd_perlayer}\n")
                f.write(f"----- Magnitude: {massiveactmaginembd_perlayer}\n")
                f.write("*" * 50 + "\n")

            f.write("*" * 50 + "\n")
            f.write("\n")
            data_i += 1

        embd_location_firstlayer = sorted(list(embd_location_firstlayer))
        f.write("=" * 50 + "\n")
        f.write("Summary\n")
        f.write(f"First Layer with Massive Activations: {first_layer}\n")
        f.write(
            f"Embedding Position with Massive Activations in First Layer: {embd_location_firstlayer}\n"
        )
        f.write("Massive Activation Magnitude in First Layer:\n")
        for tokenpos, pos_dict in massiveactmagnitude_firstlayer.items():
            f.write(f"-- Token Position {tokenpos}\n")
            for pos, mag_list in pos_dict.items():
                f.write(
                    f"---- Position {pos} | Avg: {np.mean(mag_list,axis=0)}, Std: {np.std(mag_list,axis=0)}\n"
                )
        f.write(
            f"Unique Tokens (with the Relative Position) with Massive Activations in All Layers: {tokenswithmassiveact_all_unique}\n"
        )
    return first_layer, embd_location_firstlayer, tokenswithmassiveact_all_unique

def run_intervention_analysis(
    save_dir_sub,
    model,
    tokenizer,
    layer_path,
    context_length,
    add_bos_token,
    seed,
    first_layer,
    embd_location,
):
    def _run_eval_ppl(
        key,
        massiveactposinembd_resetvaluestarting_dict=None,
        massiveactposinembd_resetvaluenonstarting_dict=None,
        write_file=None,
    ):
        ppl_dict[key] = {}
        for ds_name, (
            testseq_list,
            tokenposwithmassiveact_starting_list,
            tokenposwithmassiveact_nonstarting_list,
        ) in testseq_dict.items():
            print(f"Calculating perplexity for {key} on {ds_name}")
            nlls = []
            with torch.no_grad():
                for (
                    test_seq,
                    tokenposwithmassiveact_starting,
                    tokenposwithmassiveact_nonstarting,
                ) in tqdm(
                    zip(
                        testseq_list,
                        tokenposwithmassiveact_starting_list,
                        tokenposwithmassiveact_nonstarting_list,
                    )
                ):
                    all_hooks = []
                    if (
                        massiveactposinembd_resetvaluestarting_dict is not None
                        and massiveactposinembd_resetvaluenonstarting_dict is not None
                    ):
                        if tokenposwithmassiveact_starting:
                            hook = reset_first_massive_activations(
                                model,
                                layer_path,
                                first_layer,
                                massiveactposinembd_resetvalue_dict=massiveactposinembd_resetvaluestarting_dict,
                                token_dim=tokenposwithmassiveact_starting,
                                write_file=os.path.join(write_file, f"{ds_name}_startingtoken.txt"),
                            )
                            all_hooks.append(hook)

                        if tokenposwithmassiveact_nonstarting:
                            hook = reset_first_massive_activations(
                                model,
                                layer_path,
                                first_layer,
                                massiveactposinembd_resetvalue_dict=massiveactposinembd_resetvaluenonstarting_dict,
                                token_dim=tokenposwithmassiveact_nonstarting,
                                write_file=os.path.join(
                                    write_file, f"{ds_name}_nonstartingtoken.txt"
                                ),
                            )
                            all_hooks.append(hook)

                    lm_logits = model(test_seq).logits

                    shift_logits = lm_logits[:, :-1, :].contiguous() 
                    shift_labels = test_seq[:, 1:]

                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
                    )
                    neg_log_likelihood = loss.float() * test_seq.numel()
                    nlls.append(neg_log_likelihood)

                    for hook in all_hooks:
                        hook.remove()

            try:
                ppl = torch.exp(
                    torch.stack(nlls).sum() / (len(testseq_list) * context_length)
                ).item()

                ppl_dict[key][ds_name] = ppl
            except Exception as e:
                print(e)
                ppl_dict[key][ds_name] = None

    if os.path.exists(os.path.join(save_dir_sub, "intervention_analysis")):
        shutil.rmtree(os.path.join(save_dir_sub, "intervention_analysis"))

    ds_list = ["wikitext", "c4", "pg19"]
    ppl_dict = {}

    print("Caching test data along with the massive activations positions")
    testseq_dict = {}
    for ds_name in ds_list:
        print(f"Caching {ds_name}...")
        testseq_list = get_test_data(
            ds_name,
            tokenizer,
            add_bos_token=add_bos_token,
            seed=seed,
            seqlen=context_length,
            device="cuda",
        )
        tokenposwithmassiveact_starting_list = []
        tokenposwithmassiveact_nonstarting_list = []
        for test_seq in tqdm(testseq_list):
            massive_activation_stats = get_massive_activations_stats(
                test_seq,
                model,
                tokenizer,
                layer_path,
            )

            if massive_activation_stats is None:
                tokenposwithmassiveact = []
            else:
                (
                    _,
                    tokenposwithmassiveact,
                    _,
                    _,
                    _,
                    _,
                ) = massive_activation_stats

            tokenposwithmassiveact_starting_list.append(
                [x for x in tokenposwithmassiveact if x == 0]
            )
            tokenposwithmassiveact_nonstarting_list.append(
                [x for x in tokenposwithmassiveact if x != 0]
            )
        testseq_dict[ds_name] = (
            testseq_list,
            tokenposwithmassiveact_starting_list,
            tokenposwithmassiveact_nonstarting_list,
        )

    # Original
    print("Calculating original perplexity")
    _run_eval_ppl("original")

    # Set massive activations to zero where it first appears in the layers
    print("Intervening with zero value of massive activations")
    os.makedirs(
        os.path.join(save_dir_sub, "intervention_analysis", "set_to_zero_ppl"),
        exist_ok=True,
    )
    _run_eval_ppl(
        "set_to_zero",
        massiveactposinembd_resetvaluestarting_dict={k: 0.0 for k in embd_location},
        massiveactposinembd_resetvaluenonstarting_dict={k: 0.0 for k in embd_location},
        write_file=os.path.join(save_dir_sub, "intervention_analysis", "set_to_zero_ppl"),
    )

    # Set massive activations to mean where it first appears in the layers
    print("Intervening with mean value of massive activations")
    calibrationdata = get_data(
        tokenizer,
        nsamples=100,
        seqlen=context_length,
        add_bos_token=add_bos_token,
        device="cuda",
    )

    massive_activations_avg_dict = {}
    for valenc in tqdm(calibrationdata):
        massive_activation_stats = get_massive_activations_stats(
            valenc,
            model,
            tokenizer,
            layer_path,
        )

        if massive_activation_stats is None:
            continue
        else:
            (
                first_layer_pos_with_massive_act,
                tokenposwithmassiveact,
                massiveactposinembd,
                _,
                stats,
                _,
            ) = massive_activation_stats
            tokenposwithmassiveact = tuple(tokenposwithmassiveact)
            massiveactposinembd = tuple(massiveactposinembd)

        if first_layer_pos_with_massive_act not in massive_activations_avg_dict:
            massive_activations_avg_dict[first_layer_pos_with_massive_act] = {}

        tensor_ori = stats[f"{first_layer_pos_with_massive_act}_ori"]
        for loc in massiveactposinembd:
            if loc not in embd_location:
                continue
            if loc not in massive_activations_avg_dict[first_layer_pos_with_massive_act]:
                massive_activations_avg_dict[first_layer_pos_with_massive_act][loc] = {}
            for tokenpos in tokenposwithmassiveact:
                is_starting = tokenpos == 0
                if (
                    is_starting
                    not in massive_activations_avg_dict[first_layer_pos_with_massive_act][loc]
                ):
                    massive_activations_avg_dict[first_layer_pos_with_massive_act][loc][
                        is_starting
                    ] = []
                massive_activations_avg_dict[first_layer_pos_with_massive_act][loc][
                    is_starting
                ].append(tensor_ori[:, tokenpos, loc].cpu().item())

    embd_location_from_avg = set()
    for (
        first_layer_pos_with_massive_act,
        loc_dict,
    ) in massive_activations_avg_dict.items():
        os.makedirs(
            os.path.join(
                save_dir_sub,
                "intervention_analysis",
                f"set_to_mean_{first_layer_pos_with_massive_act}_ppl",
            ),
            exist_ok=True,
        )
        massiveactposinembd_resetvaluestarting_dict = {}
        massiveactposinembd_resetvaluenonstarting_dict = {}
        for loc, massive_activations_list_dict in loc_dict.items():
            embd_location_from_avg.add(loc)
            if True in massive_activations_list_dict:
                massiveactposinembd_resetvaluestarting_dict[loc] = np.mean(
                    massive_activations_list_dict[True]
                )
            if False in massive_activations_list_dict:
                massiveactposinembd_resetvaluenonstarting_dict[loc] = np.mean(
                    massive_activations_list_dict[False]
                )
        _run_eval_ppl(
            f"set_to_mean_{first_layer_pos_with_massive_act}",
            massiveactposinembd_resetvaluestarting_dict=massiveactposinembd_resetvaluestarting_dict,
            massiveactposinembd_resetvaluenonstarting_dict=massiveactposinembd_resetvaluenonstarting_dict,
            write_file=os.path.join(
                save_dir_sub,
                "intervention_analysis",
                f"set_to_mean_{first_layer_pos_with_massive_act}_ppl",
            ),
        )

    missing_embd_location_from_avg = [x for x in embd_location if x not in embd_location_from_avg]
    with open(os.path.join(save_dir_sub, "intervention_analysis", "results_ppl.txt"), "w") as f:
        f.write("Perplexity Results\n")
        for key, ppl_ds in ppl_dict.items():
            f.write(f"{key}\n")
            for ds_name, ppl in ppl_ds.items():
                f.write(f"{ds_name}: {ppl}\n")
            f.write("\n")
        if missing_embd_location_from_avg:
            f.write("Missing Embedding Location When Intervening with Mean\n")
            f.write(f"{missing_embd_location_from_avg}\n")

    return ppl_dict