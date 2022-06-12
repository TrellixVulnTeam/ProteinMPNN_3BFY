import json, time, os, sys, glob

import gradio as gr

sys.path.append("/home/user/app/ProteinMPNN/vanilla_proteinmpnn")

import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import os.path
from protein_mpnn_utils import (
    loss_nll,
    loss_smoothed,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
    cat_neighbors_nodes,
    _scores,
    _S_to_seq,
    tied_featurize,
    parse_PDB,
)
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
import plotly.express as px
import urllib
import jax.numpy as jnp
import tensorflow as tf

if "/home/user/app/af_backprop" not in sys.path:
    sys.path.append("/home/user/app/af_backprop")

from utils import *

# import libraries
import colabfold as cf
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.model import data, config, model
from alphafold.common import residue_constants


import plotly.graph_objects as go
import ray

import re

import numpy as np
import jax

tf.config.set_visible_devices([], "GPU")


def chain_break(idx_res, Ls, length=200):
    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i :] += length
        L_prev += L_i
    return idx_res


def setup_model(seq, model_name="model_1_ptm"):

    # setup model
    cfg = config.model_config("model_1_ptm")
    cfg.model.num_recycle = 0
    cfg.data.common.num_recycle = 0
    cfg.data.eval.max_msa_clusters = 1
    cfg.data.common.max_extra_msa = 1
    cfg.data.eval.masked_msa_replace_fraction = 0
    cfg.model.global_config.subbatch_size = None
    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=".")
    model_runner = model.RunModel(cfg, model_params, is_training=False)
    Ls = [len(s) for s in seq.split("/")]

    seq = re.sub("[^A-Z]", "", seq.upper())
    length = len(seq)
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=seq, description="none", num_res=length
        ),
        **pipeline.make_msa_features(msas=[[seq]], deletion_matrices=[[[0] * length]]),
    }
    feature_dict["residue_index"] = chain_break(feature_dict["residue_index"], Ls)
    inputs = model_runner.process_features(feature_dict, random_seed=0)

    def runner(seq, opt):
        # update sequence
        inputs = opt["inputs"]
        inputs.update(opt["prev"])
        update_seq(seq, inputs)
        update_aatype(inputs["target_feat"][..., 1:], inputs)

        # mask prediction
        mask = seq.sum(-1)
        inputs["seq_mask"] = inputs["seq_mask"].at[:].set(mask)
        inputs["msa_mask"] = inputs["msa_mask"].at[:].set(mask)
        inputs["residue_index"] = jnp.where(mask == 1, inputs["residue_index"], 0)

        # get prediction
        key = jax.random.PRNGKey(0)
        outputs = model_runner.apply(opt["params"], key, inputs)

        prev = {
            "init_msa_first_row": outputs["representations"]["msa_first_row"][None],
            "init_pair": outputs["representations"]["pair"][None],
            "init_pos": outputs["structure_module"]["final_atom_positions"][None],
        }

        aux = {
            "final_atom_positions": outputs["structure_module"]["final_atom_positions"],
            "final_atom_mask": outputs["structure_module"]["final_atom_mask"],
            "plddt": get_plddt(outputs),
            "pae": get_pae(outputs),
            "inputs": inputs,
            "prev": prev,
        }
        return aux

    return jax.jit(runner), {"inputs": inputs, "params": model_params}


def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted(
            [item[-1:] for item in list(result) if item[:9] == "seq_chain"]
        )  # A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i]  # needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result["name"]] = tied_positions_list
    return my_dict


def renumber(struc):
    """Renumber residues consecutively and remove all hetero residues"""
    resid = 0
    residue_to_remove = []
    chain_to_remove = []
    for model in struc:
        for chain in model:
            for i, residue in enumerate(chain.get_residues()):
                res_id = list(residue.id)
                res_id[1] = resid
                resid += 1
                residue.id = tuple(res_id)
                if residue.id[0] != " ":
                    residue_to_remove.append((chain.id, residue.id))
            if len(chain) == 0:
                chain_to_remove.append(chain.id)
    for residue in residue_to_remove:
        struc[0][residue[0]].detach_child(residue[1])

    for chain in chain_to_remove:
        model.detach_child(chain)
    return struc


def align_structures(pdb1, pdb2, lenRes):
    """Take two structure and superimpose pdb1 on pdb2"""
    import Bio.PDB

    # We use all residues
    atoms_to_be_aligned = range(0, lenRes)

    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    # Get the structures
    ref_structure = pdb_parser.get_structure("samle", pdb2)
    sample_structure = renumber(pdb_parser.get_structure("reference", pdb1))
    # Use the first model in the pdb-files for alignment
    ref_model = ref_structure[0]
    sample_model = sample_structure[0]
    # Make a list of the atoms (in the structures) to align.
    ref_atoms = []
    sample_atoms = []
    # Iterate of all chains in the model in order to find all residues
    for ref_chain in ref_model:
        # Iterate of all residues in each model in order to find proper atoms
        for ref_res in ref_chain:
            # Check if residue number ( .get_id() ) is in the list
            if ref_res.get_id()[1] in atoms_to_be_aligned:
                # Append CA atom to list
                ref_atoms.append(ref_res["CA"])
    for sample_chain in sample_model:
        for sample_res in sample_chain:
            if sample_res.get_id()[1] in atoms_to_be_aligned:
                sample_atoms.append(sample_res["CA"])

    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())

    io = Bio.PDB.PDBIO()
    io.set_structure(sample_structure)
    io.save(f"{pdb1}_aligned.pdb")
    return super_imposer.rms, f"{pdb1}_aligned.pdb"


def save_pdb(outs, filename, LEN):
    """save pdb coordinates"""
    p = {
        "residue_index": outs["inputs"]["residue_index"][0][:LEN],
        "aatype": outs["inputs"]["aatype"].argmax(-1)[0][:LEN],
        "atom_positions": outs["final_atom_positions"][:LEN],
        "atom_mask": outs["final_atom_mask"][:LEN],
    }
    b_factors = 100.0 * outs["plddt"][:LEN, None] * p["atom_mask"]
    p = protein.Protein(**p, b_factors=b_factors)
    pdb_lines = protein.to_pdb(p)
    with open(filename, "w") as f:
        f.write(pdb_lines)
    print(os.listdir(), os.getcwd())


@ray.remote(num_gpus=1, max_calls=1)
def run_alphafold(sequence):
    recycles = 3
    RUNNER, OPT = setup_model(sequence)

    SEQ = re.sub("[^A-Z]", "", sequence.upper())
    MAX_LEN = len(SEQ)
    LEN = len(SEQ)

    x = np.array([residue_constants.restype_order.get(aa, -1) for aa in SEQ])
    x = np.pad(x, [0, MAX_LEN - LEN], constant_values=-1)
    x = jax.nn.one_hot(x, 20)

    OPT["prev"] = {
        "init_msa_first_row": np.zeros([1, MAX_LEN, 256]),
        "init_pair": np.zeros([1, MAX_LEN, MAX_LEN, 128]),
        "init_pos": np.zeros([1, MAX_LEN, 37, 3]),
    }

    positions = []
    plddts = []
    for r in range(recycles + 1):
        outs = RUNNER(x, OPT)
        outs = jax.tree_map(lambda x: np.asarray(x), outs)
        positions.append(outs["prev"]["init_pos"][0, :LEN])
        plddts.append(outs["plddt"][:LEN])
        OPT["prev"] = outs["prev"]
        if recycles > 0:
            print(r, plddts[-1].mean())
    save_pdb(outs, "out.pdb", LEN)
    num_res = int(outs["inputs"]["aatype"][0].sum())
    return outs["plddt"], outs["pae"], num_res


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model_name = "v_48_020"  # ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030, v_32_002, v_32_010; v_32_020, v_32_030; v_48_010=version with 48 edges 0.10A noise
backbone_noise = 0.00  # Standard deviation of Gaussian noise to add to backbone atoms

path_to_model_weights = (
    "/home/user/app/ProteinMPNN/vanilla_proteinmpnn/vanilla_model_weights"
)
hidden_dim = 128
num_layers = 3
model_folder_path = path_to_model_weights
if model_folder_path[-1] != "/":
    model_folder_path = model_folder_path + "/"
checkpoint_path = model_folder_path + f"{model_name}.pt"

checkpoint = torch.load(checkpoint_path, map_location=device)

noise_level_print = checkpoint["noise_level"]

model = ProteinMPNN(
    num_letters=21,
    node_features=hidden_dim,
    edge_features=hidden_dim,
    hidden_dim=hidden_dim,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    augment_eps=backbone_noise,
    k_neighbors=checkpoint["num_edges"],
)
model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def get_pdb(pdb_code="", filepath=""):
    if pdb_code is None or pdb_code == "":
        return filepath.name
    else:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"


def update(inp, file, designed_chain, fixed_chain, homomer, num_seqs, sampling_temp):
    pdb_path = get_pdb(pdb_code=inp, filepath=file)
    if designed_chain == "":
        designed_chain_list = []
    else:
        designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

    if fixed_chain == "":
        fixed_chain_list = []
    else:
        fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

    chain_list = list(set(designed_chain_list + fixed_chain_list))
    num_seq_per_target = num_seqs
    save_score = 0  # 0 for False, 1 for True; save score=-log_prob to npy files
    save_probs = (
        0  # 0 for False, 1 for True; save MPNN predicted probabilites per position
    )
    score_only = 0  # 0 for False, 1 for True; score input backbone-sequence pairs
    conditional_probs_only = 0  # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
    conditional_probs_only_backbone = 0  # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

    batch_size = 1  # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
    max_length = 20000  # Max sequence length

    out_folder = "."  # Path to a folder to output sequences, e.g. /home/out/
    jsonl_path = ""  # Path to a folder with parsed pdb into jsonl
    omit_AAs = "X"  # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

    pssm_multi = 0.0  # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
    pssm_threshold = 0.0  # A value between -inf + inf to restric per position AAs
    pssm_log_odds_flag = 0  # 0 for False, 1 for True
    pssm_bias_flag = 0  # 0 for False, 1 for True

    folder_for_outputs = out_folder

    NUM_BATCHES = num_seq_per_target // batch_size
    BATCH_COPIES = batch_size
    temperatures = [sampling_temp]
    omit_AAs_list = omit_AAs
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"

    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    bias_AA_dict = None

    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    ###############################################################
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(
        pdb_dict_list, truncate=None, max_length=max_length
    )
    if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
    else:
        tied_positions_dict = None
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]["name"]] = (designed_chain_list, fixed_chain_list)
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            (
                X,
                S,
                mask,
                lengths,
                chain_M,
                chain_encoding_all,
                chain_list_list,
                visible_list_list,
                masked_list_list,
                masked_chain_length_list_list,
                chain_M_pos,
                omit_AA_mask,
                residue_idx,
                dihedral_mask,
                tied_pos_list_of_lists_list,
                pssm_coef,
                pssm_bias,
                pssm_log_odds_all,
                bias_by_res_all,
                tied_beta,
            ) = tied_featurize(
                batch_clones,
                device,
                chain_id_dict,
                fixed_positions_dict,
                omit_AA_dict,
                tied_positions_dict,
                pssm_dict,
                bias_by_res_dict,
            )
            pssm_log_odds_mask = (
                pssm_log_odds_all > pssm_threshold
            ).float()  # 1.0 for true, 0.0 for false
            name_ = batch_clones[0]["name"]

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(
                X,
                S,
                mask,
                chain_M * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                randn_1,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()
            message = ""
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    if tied_positions_dict == None:
                        sample_dict = model.sample(
                            X,
                            randn_2,
                            S,
                            chain_M,
                            chain_encoding_all,
                            residue_idx,
                            mask=mask,
                            temperature=temp,
                            omit_AAs_np=omit_AAs_np,
                            bias_AAs_np=bias_AAs_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask,
                            pssm_coef=pssm_coef,
                            pssm_bias=pssm_bias,
                            pssm_multi=pssm_multi,
                            pssm_log_odds_flag=bool(pssm_log_odds_flag),
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=bool(pssm_bias_flag),
                            bias_by_res=bias_by_res_all,
                        )
                        S_sample = sample_dict["S"]
                    else:
                        sample_dict = model.tied_sample(
                            X,
                            randn_2,
                            S,
                            chain_M,
                            chain_encoding_all,
                            residue_idx,
                            mask=mask,
                            temperature=temp,
                            omit_AAs_np=omit_AAs_np,
                            bias_AAs_np=bias_AAs_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask,
                            pssm_coef=pssm_coef,
                            pssm_bias=pssm_bias,
                            pssm_multi=pssm_multi,
                            pssm_log_odds_flag=bool(pssm_log_odds_flag),
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=bool(pssm_bias_flag),
                            tied_pos=tied_pos_list_of_lists_list[0],
                            tied_beta=tied_beta,
                            bias_by_res=bias_by_res_all,
                        )
                        # Compute scores
                        S_sample = sample_dict["S"]
                    log_probs = model(
                        X,
                        S_sample,
                        mask,
                        chain_M * chain_M_pos,
                        residue_idx,
                        chain_encoding_all,
                        randn_2,
                        use_input_decoding_order=True,
                        decoding_order=sample_dict["decoding_order"],
                    )
                    mask_for_loss = mask * chain_M * chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(
                            torch.sum(
                                torch.nn.functional.one_hot(S[b_ix], 21)
                                * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                                axis=-1,
                            )
                            * mask_for_loss[b_ix]
                        ) / torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j == 0 and temp == temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(
                                list(np.array(list_of_AAs)[np.argsort(masked_list)])
                            )
                            l0 = 0
                            for mc_length in list(
                                np.array(masked_chain_length_list)[
                                    np.argsort(masked_list)
                                ]
                            )[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                                l0 += 1
                            sorted_masked_chain_letters = np.argsort(
                                masked_list_list[0]
                            )
                            print_masked_chains = [
                                masked_list_list[0][i]
                                for i in sorted_masked_chain_letters
                            ]
                            sorted_visible_chain_letters = np.argsort(
                                visible_list_list[0]
                            )
                            print_visible_chains = [
                                visible_list_list[0][i]
                                for i in sorted_visible_chain_letters
                            ]
                            native_score_print = np.format_float_positional(
                                np.float32(native_score.mean()),
                                unique=False,
                                precision=4,
                            )
                            line = ">{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n".format(
                                name_,
                                native_score_print,
                                print_visible_chains,
                                print_masked_chains,
                                model_name,
                                native_seq,
                            )
                            message += f"{line}\n"
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(
                            list(np.array(list_of_AAs)[np.argsort(masked_list)])
                        )
                        l0 = 0
                        for mc_length in list(
                            np.array(masked_chain_length_list)[np.argsort(masked_list)]
                        )[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + "/" + seq[l0:]
                            l0 += 1
                        score_print = np.format_float_positional(
                            np.float32(score), unique=False, precision=4
                        )
                        seq_rec_print = np.format_float_positional(
                            np.float32(seq_recovery_rate.detach().cpu().numpy()),
                            unique=False,
                            precision=4,
                        )
                        line = (
                            ">T={}, sample={}, score={}, seq_recovery={}\n{}\n".format(
                                temp, b_ix, score_print, seq_rec_print, seq
                            )
                        )
                        message += f"{line}\n"

    all_probs_concat = np.concatenate(all_probs_list)
    all_log_probs_concat = np.concatenate(all_log_probs_list)
    np.savetxt("all_probs_concat.csv", all_probs_concat.mean(0).T, delimiter=",")
    np.savetxt(
        "all_log_probs_concat.csv",
        np.exp(all_log_probs_concat).mean(0).T,
        delimiter=",",
    )
    S_sample_concat = np.concatenate(S_sample_list)
    fig = px.imshow(
        np.exp(all_log_probs_concat).mean(0).T,
        labels=dict(x="positions", y="amino acids", color="probability"),
        y=list(alphabet),
        template="simple_white",
    )
    fig.update_xaxes(side="top")

    fig_tadjusted = px.imshow(
        all_probs_concat.mean(0).T,
        labels=dict(x="positions", y="amino acids", color="probability"),
        y=list(alphabet),
        template="simple_white",
    )

    fig_tadjusted.update_xaxes(side="top")

    return (
        message,
        fig,
        fig_tadjusted,
        gr.File.update(value="all_log_probs_concat.csv", visible=True),
        gr.File.update(value="all_probs_concat.csv", visible=True),
        pdb_path,
    )


def update_AF(startsequence, pdb):

    # # run alphafold using ray
    plddts, pae, num_res = ray.get(run_alphafold.remote(startsequence))
    x = np.arange(10)

    plotAF_plddt = go.Figure(
        data=go.Scatter(
            x=np.arange(len(plddts)),
            y=plddts,
            hovertemplate="<i>pLDDT</i>: %{y:.2f} <br><i>Residue index:</i> %{x}",
        )
    )
    plotAF_plddt.update_layout(
        title="pLDDT",
        xaxis_title="Residue index",
        yaxis_title="pLDDT",
        height=500,
        template="simple_white",
    )

    plotAF_pae = px.imshow(
        pae,
        labels=dict(x="Scored residue", y="Aligned residue", color=""),
    )
    plotAF_pae.update_layout(title="Predicted Aligned Error", template="simple_white")

    return molecule(pdb, "af_backprop/out.pdb", num_res), plotAF_plddt, plotAF_pae


def read_mol(molpath):
    with open(molpath, "r") as fp:
        lines = fp.readlines()
    mol = ""
    for l in lines:
        mol += l
    return mol


def molecule(pdb, afpdb, num_res):

    rms, aligned_pdb = align_structures(pdb, afpdb, num_res)

    mol = read_mol(pdb)
    pred_mol = read_mol(aligned_pdb)
    x = (
        """<!DOCTYPE html>
        <html>
        <head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
     <link rel="stylesheet" href="https://unpkg.com/flowbite@1.4.5/dist/flowbite.min.css" />
    <style>
    body{
        font-family:sans-serif
    }
.mol-container {
  width: 100%;
  height: 800px;
  position: relative;
}
.space-x-2 > * + *{
    margin-left: 0.5rem;
}
.p-1{
    padding:0.5rem;
}
.flex{
    display:flex;
    align-items: center;
}
.w-4{
    width:1rem;
}
.h-4{
    height:1rem;
}
.mt-4{
    margin-top:1rem;
}
select{
    background-image:None;
}
</style>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>  
   
    <div id="container" class="mol-container"></div>
    <div class="flex">
        <div class="px-4">
        <label for="sidechain" class="relative inline-flex items-center mb-4 cursor-pointer ">
            <input  id="sidechain"type="checkbox" class="sr-only peer">
            <div class="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            <span class="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300">Show side chains</span>
          </label>
        </div>
 <button type="button" class="text-gray-900 bg-white hover:bg-gray-100 border border-gray-200 focus:ring-4 focus:outline-none focus:ring-gray-100 font-medium rounded-lg text-sm px-5 py-2.5 text-center inline-flex items-center dark:focus:ring-gray-600 dark:bg-gray-800 dark:border-gray-700 dark:text-white dark:hover:bg-gray-700 mr-2 mb-2" id="download">
                    <svg class="w-6 h-6 mr-2 -ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    Download predicted structure
                  </button>
            </div>       
<div class="text-sm">
                            <div class="font-medium mt-4"><b>AlphaFold model confidence:</b></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4"
                                    style="background-color: rgb(0, 83, 214);">&nbsp;</span><span class="legendlabel">Very high
                                    (pLDDT &gt; 90)</span></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4"
                                    style="background-color: rgb(101, 203, 243);">&nbsp;</span><span class="legendlabel">Confident
                                    (90 &gt; pLDDT &gt; 70)</span></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4"
                                    style="background-color: rgb(255, 219, 19);">&nbsp;</span><span class="legendlabel">Low (70 &gt;
                                    pLDDT &gt; 50)</span></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4"
                                    style="background-color: rgb(255, 125, 69);">&nbsp;</span><span class="legendlabel">Very low
                                    (pLDDT &lt; 50)</span></div>
                            <div class="row column legendDesc"> AlphaFold produces a per-residue confidence
                                score (pLDDT) between 0 and 100. Some regions below 50 pLDDT may be unstructured in isolation.
                            </div>
                        </div>
            <script>
            let viewer = null;
            let voldata = null;
            $(document).ready(function () {
                let element = $("#container");
                let config = { backgroundColor: "white" };
                viewer = $3Dmol.createViewer( element, config );
                viewer.ui.initiateUI();
                let data = `"""
        + pred_mol
        + """`  
                let pdb = `"""
        + mol
        + """`  
                viewer.addModel( data, "pdb" );
                viewer.addModel( pdb, "pdb" );
                //AlphaFold code from https://gist.github.com/piroyon/30d1c1099ad488a7952c3b21a5bebc96
                let colorAlpha = function (atom) {
                    if (atom.b < 50) {
                        return "OrangeRed";
                    } else if (atom.b < 70) {
                        return "Gold";
                    } else if (atom.b < 90) {
                        return "MediumTurquoise";
                    } else {
                        return "Blue";
                    }
                };
                viewer.setStyle({}, { cartoon: { colorfunc: colorAlpha } });
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(0.8, 2000);
                viewer.getModel(0).setHoverable({}, true,
                    function (atom, viewer, event, container) {
                        console.log(atom)
                        if (!atom.label) {
                            atom.label = viewer.addLabel(atom.resn+atom.resi+" pLDDT=" + atom.b, { position: atom, backgroundColor: "mintcream", fontColor: "black" });
                        }
                    },
                    function (atom, viewer) {
                        if (atom.label) {
                            viewer.removeLabel(atom.label);
                            delete atom.label;
                        }
                    }
                );
                $("#sidechain").change(function () {
                    if (this.checked) {
                        BB = ["C", "O", "N"]
                        viewer.setStyle( {"and": [{resn: ["GLY", "PRO"], invert: true},{atom: BB, invert: true},]},{stick: {colorscheme: "WhiteCarbon", radius: 0.3}, cartoon: { colorfunc: colorAlpha }});
                        viewer.render()
                    } else {
                        viewer.setStyle({cartoon: { colorfunc: colorAlpha }});
                        viewer.render()
                    }
                });
                $("#download").click(function () {
                    download(\""""
        + aligned_pdb
        + """\", data);
                })
        });
        function download(filename, text) {
            var element = document.createElement("a");
            element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(text));
            element.setAttribute("download", filename);
            element.style.display = "none";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }
        </script>
        </body></html>"""
    )

    return f"""<iframe style="width: 800px; height: 1200px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""


def set_examples(example):
    label, inp, designed_chain, fixed_chain, homomer, num_seqs, sampling_temp = example
    return [
        label,
        inp,
        designed_chain,
        fixed_chain,
        homomer,
        gr.Slider.update(value=num_seqs),
        gr.Radio.update(value=sampling_temp),
    ]


proteinMPNN = gr.Blocks()

with proteinMPNN:
    gr.Markdown("# ProteinMPNN")
    gr.Markdown(
        """This model takes as input a protein structure and based on its backbone predicts new sequences that will fold into that backbone. 
        Optionally, we can run AlphaFold2 on the predicted sequence to check whether the predicted sequences adopt the same backbone (WIP). 
        """
    )
    gr.Markdown("![](https://simonduerr.eu/ProteinMPNN.png)")

    with gr.Tabs():
        with gr.TabItem("Input"):
            inp = gr.Textbox(
                placeholder="PDB Code or upload file below", label="Input structure"
            )
            file = gr.File(file_count="single", type="file")

        with gr.TabItem("Settings"):
            with gr.Row():
                designed_chain = gr.Textbox(value="A", label="Designed chain")
                fixed_chain = gr.Textbox(
                    placeholder="Use commas to fix multiple chains", label="Fixed chain"
                )
            with gr.Row():
                num_seqs = gr.Slider(
                    minimum=1, maximum=50, value=1, step=1, label="Number of sequences"
                )
                sampling_temp = gr.Radio(
                    choices=[0.1, 0.15, 0.2, 0.25, 0.3],
                    value=0.1,
                    label="Sampling temperature",
                )
            with gr.Row():
                homomer = gr.Checkbox(value=False, label="Homomer?")
                gr.Markdown(
                    "for correct symmetric tying lenghts of homomer chains should be the same"
                )

        btn = gr.Button("Run")
    label = gr.Textbox(label="Label", visible=False)
    examples = gr.Dataset(
        components=[
            label,
            inp,
            designed_chain,
            fixed_chain,
            homomer,
            num_seqs,
            sampling_temp,
        ],
        samples=[
            ["Homomer design", "1O91", "A,B,C", "", True, 2, 0.1],
            ["Monomer design", "6MRR", "A", "", False, 2, 0.1],
            ["Redesign of Homomer to Heteromer", "3HTN", "A,B", "C", False, 2, 0.1],
        ],
    )
    gr.Markdown(
        """ Sampling temperature for amino acids, `T=0.0` means taking argmax, `T>>1.0` means sample randomly. Suggested values `0.1, 0.15, 0.2, 0.25, 0.3`. Higher values will lead to more diversity.
    """
    )

    gr.Markdown("# Output")

    with gr.Tabs():
        with gr.TabItem("Designed sequences"):
            out = gr.Textbox(label="Status")

        with gr.TabItem("Amino acid probabilities"):
            plot = gr.Plot()
            all_log_probs = gr.File(visible=False)
        with gr.TabItem("T adjusted probabilities"):
            gr.Markdown("Sampling temperature adjusted amino acid probabilties")
            plot_tadjusted = gr.Plot()
            all_probs = gr.File(visible=False)
        with gr.TabItem("Structure validation w/ AF2"):
            # gr.Markdown("Coming soon")
            with gr.Row():
                chosen_seq = gr.Textbox(
                    label="Copy and paste a sequence for validation"
                )
                btnAF = gr.Button("Run AF2 on sequence")
            mol = gr.HTML()
            with gr.Row():
                plotAF_plddt = gr.Plot(label="pLDDT")
                plotAF_pae = gr.Plot(label="PAE")
    file = gr.Variable()
    btn.click(
        fn=update,
        inputs=[
            inp,
            file,
            designed_chain,
            fixed_chain,
            homomer,
            num_seqs,
            sampling_temp,
        ],
        outputs=[out, plot, plot_tadjusted, all_log_probs, all_probs, file],
    )
    btnAF.click(
        fn=update_AF,
        inputs=[chosen_seq, file],
        outputs=[mol, plotAF_plddt, plotAF_pae],
    )
    examples.click(fn=set_examples, inputs=examples, outputs=examples.components)
    gr.Markdown(
        """Citation: **Robust deep learning based protein sequence design using ProteinMPNN** <br>
Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Alexis Courbet, Robbert J. de Haas, Neville Bethel, Philip J. Y. Leung, Timothy F. Huddy, Sam Pellock, Doug Tischer, Frederick Chan, Brian Koepnick, Hannah Nguyen, Alex Kang, Banumathi Sankaran, Asim Bera, Neil P. King, David Baker <br>
bioRxiv 2022.06.03.494563; doi: [10.1101/2022.06.03.494563](https://doi.org/10.1101/2022.06.03.494563) <br><br> Server built by [@simonduerr](https://twitter.com/simonduerr) and hosted by Huggingface"""
    )


ray.init(runtime_env={"working_dir": "./"})

proteinMPNN.launch()
