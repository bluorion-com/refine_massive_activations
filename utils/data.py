import random

import torch
import numpy as np
from datasets import load_dataset

def get_data(tokenizer, nsamples=50, seqlen=4096, add_bos_token=False, device=None):
    valdata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", trust_remote_code=True)

    num_seq = len(valdata["train"])
    seq_indices = np.random.choice(num_seq, 500, replace=False).tolist()
    seq_list = []
    for seq_ind in seq_indices:
        seq_list.append(valdata["train"][seq_ind]["text"])

    testseq_list = tokenize_data(
        seq_list,
        tokenizer,
        nsamples,
        seqlen=seqlen,
        add_bos_token=add_bos_token,
        device=device,
    )
    return testseq_list

def get_test_data(dataset_name, tokenizer, add_bos_token=False, seed=42, seqlen=2048, device=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if dataset_name == "wikitext":
        return get_wikitext(tokenizer, seqlen=seqlen, add_bos_token=add_bos_token, device=device)
    elif dataset_name == "c4":
        return get_c4(tokenizer, seqlen=seqlen, add_bos_token=add_bos_token, device=device)
    elif dataset_name == "pg19":
        return get_pg19(tokenizer, seqlen=seqlen, add_bos_token=add_bos_token, device=device)

def tokenize_data(seq_list, tokenizer, nsamples, seqlen=4096, add_bos_token=False, device=None):
    testenc = tokenizer("\n\n".join(seq_list), add_special_tokens=False).input_ids

    testseq_list = []
    for i in range(nsamples):
        if add_bos_token:
            test_seq = torch.tensor(
                [[tokenizer.bos_token_id] + testenc[(i * seqlen) : ((i + 1) * seqlen - 1)]]
            ).to(device)
        else:
            test_seq = torch.tensor([testenc[(i * seqlen) : ((i + 1) * seqlen)]]).to(device)
        if test_seq.nelement() < seqlen:
            break
        testseq_list.append(test_seq.reshape(1, seqlen))
    return testseq_list

def get_wikitext(tokenizer, seqlen=2048, add_bos_token=False, device=None):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    seq_list = testdata["text"]
    testenc = tokenizer("\n\n".join(seq_list), add_special_tokens=False)
    testenc = testenc.input_ids

    testseq_list = []
    nsamples = len(testenc) // seqlen

    for i in range(nsamples):
        if add_bos_token:
            testenc_cur = torch.tensor(
                [[tokenizer.bos_token_id] + testenc[(i * seqlen) : ((i + 1) * seqlen - 1)]]
            ).to(device)

        else:
            testenc_cur = torch.tensor([testenc[(i * seqlen) : ((i + 1) * seqlen)]]).to(device)
        if testenc_cur.nelement() < seqlen:
            break
        testseq_list.append(testenc_cur.reshape(1, seqlen))
    return testseq_list


def get_pg19(tokenizer, seqlen=2048, add_bos_token=False, device=None):
    valdata = load_dataset("emozilla/pg19", split="validation")
    seq_list = valdata[:5]["text"]

    testseq_list = []
    valenc = tokenizer(" ".join(seq_list), add_special_tokens=False).input_ids
    for i in range(100):
        if add_bos_token:
            testseq = torch.tensor(
                [[tokenizer.bos_token_id] + valenc[(i * seqlen) : ((i + 1) * seqlen - 1)]]
            ).to(device)

        else:
            testseq = torch.tensor([valenc[(i * seqlen) : ((i + 1) * seqlen)]]).to(device)
        if testseq.nelement() < seqlen:
            break
        testseq_list.append(testseq)
    return testseq_list


def get_c4(tokenizer, seqlen=2048, add_bos_token=False, device=None):
    valdata = load_dataset("NeelNanda/c4-10k")
    seq_list = valdata["train"][:5000]["text"]

    testseq_list = []
    valenc = tokenizer(" ".join(seq_list), add_special_tokens=False).input_ids
    for i in range(100):
        if add_bos_token:
            testseq = torch.tensor(
                [[tokenizer.bos_token_id] + valenc[(i * seqlen) : ((i + 1) * seqlen - 1)]]
            ).to(device)

        else:
            testseq = torch.tensor([valenc[(i * seqlen) : ((i + 1) * seqlen)]]).to(device)
        if testseq.nelement() < seqlen:
            break
        testseq_list.append(testseq)
    return testseq_list
