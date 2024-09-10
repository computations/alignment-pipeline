import pathlib
import os
import numpy
import json
import functools
import csv
import shutil
import collections

from utils import utils, models, jplace

import warnings


with warnings.catch_warnings(action="ignore"):
    import ete3


workdir: config["prefix"]


wildcard_constraints:
    tree_iter="[0-9]+",
    pruning_iter="[0-9]+",
    damage_iter="[0-9]+",


def expand_iter_list(config_base):
    ret_list = []
    for i in range(len(config_base)):
        iter_config = config_base[i]
        iters = iter_config["iters"] if "iters" in iter_config else 1
        ret_list += [iter_config] * iters
    return ret_list


def get_program_path(program):
    if program in config:
        return os.path.abspath(os.path.expanduser(config[program]))
    return shutil.which(program)


config["exp_trees"] = expand_iter_list(config["trees"])
config["exp_models"] = expand_iter_list(config["models"])

pygargammel = get_program_path("pygargammel")

tools = ["muscle", "mafft", "clustalo", "hmmer"]

csv_fields = [
    "taxa",
    "start",
    "end",
    "formatted_name",
    "nd",
    "e_nd",
    "aligner",
    "ds",
    "ss",
    "nf",
    "ov",
    "tree_iter",
    "pruning_iter",
    "damage_iter",
]


def make_intermediat_results_list():
    files = []
    for ti, tv in enumerate(config["exp_trees"]):
        tree_path = pathlib.Path("t_" + str(ti))
        for pi in range(tv['tree']["prunings"]):
            pruning_path = tree_path / ("p_" + str(pi))
            for di, dv in enumerate(config["exp_models"]):
                damage_path = (
                    pruning_path / ("d_" + str(di)) / "epa-ng" / "distances.csv"
                )
                files.append(damage_path)
    return [str(f) for f in files]


def make_results_list():
    files = ["distances.csv"]
    return files


rule all:
    input:
        "html/plots.html",


rule python_notebook:
    input:
        damage_fasta="distances.csv",
    log:
        notebook="notebooks/plots.py.ipynb",
    conda:
        "envs/jupyter.yaml"
    notebook:
        "notebooks/plots.py.ipynb"


rule r_notebook:
    input:
        damage_fasta="distances.csv",
    log:
        notebook="notebooks/plots.r.ipynb",
    conda:
        "envs/jupyter.yaml"
    notebook:
        "notebooks/plots.r.ipynb"


rule html:
    input:
        "notebooks/plots.r.ipynb",
    output:
        "html/plots.html",
    conda:
        "envs/jupyter.yaml"
    shell:
        "jupyter nbconvert --to html --output-dir . --output {output} {input}"


rule make_tree:
    output:
        tree="t_{tree_iter}/alisim/tree.nwk",
    params:
        taxa=lookup(dpath="exp_trees/{tree_iter}/tree/taxa", within=config),
    conda:
        "envs/alisim.yaml"
    shell:
        (
            "iqtree " + "-r {params.taxa} "
            "{output.tree}; "
            "rm {output.tree}.log"
        )


rule copy_tree:
    output:
        tree="t_{tree_iter}/tree.nwk",
    input:
        tree=branch(
            lookup(dpath="exp_trees/{tree_iter}/tree/type", within=config),
            cases={
                "Simulate": "t_{tree_iter}/alisim/tree.nwk",
                "File": lookup(
                    dpath="exp_trees/{tree_iter}/tree/filename", within=config
                ),
            },
        ),
    run:
        p1 = pathlib.Path(input.tree).resolve()
        p2 = pathlib.Path(output.tree).resolve()
        p2.symlink_to(p1)


rule simulate_alignment:
    input:
        tree="t_{tree_iter}/tree.nwk",
    output:
        alignment="t_{tree_iter}/alisim/align.fasta",
    log:
        "t_{tree_iter}/alisim/alisim.log",
    params:
        length=lookup(
            dpath="exp_trees/{tree_iter}/alignment/length", within=config
        ),
        model=lookup(
            dpath="exp_trees/{tree_iter}/alignment/model", within=config
        ),
    conda:
        "envs/alisim.yaml"
    shell:
        "iqtree "
        +"--alisim {output.alignment} "
        +"-m {params.model} "
        +"-t {input.tree} "
        +"--out-format fasta "
        +"--length {params.length} "
        +"&> /dev/null;"
        +"mv {output.alignment}.fa {output.alignment};"
        +"mv {input.tree}.log {log}"


rule copy_alignment:
    output:
        align="t_{tree_iter}/align.fasta",
    input:
        align=branch(
            lookup(
                dpath="exp_trees/{tree_iter}/alignment/type", within=config
            ),
            cases={
                "File": lookup(
                    dpath="exp_trees/{tree_iter}/alignment/filename",
                    within=config,
                ),
                "Simulate": "t_{tree_iter}/alisim/align.fasta",
            },
        ),
    run:
        p1 = pathlib.Path(input.align).resolve()
        p2 = pathlib.Path(output.align).resolve()
        p2.symlink_to(p1)


rule make_pruning:
    input:
        tree="t_{tree_iter}/tree.nwk",
    output:
        tree="t_{tree_iter}/p_{pruning_iter}/pruned_tree.nwk",
        json="t_{tree_iter}/p_{pruning_iter}/pruning_info.json",
    retries: 10
    run:
        base_tree = ete3.Tree(open(input.tree).read())
        taxa_count = len(base_tree)
        subtrees = []

        min_taxa_subtree = (
            config["min_taxa_subtree"] if "min_taxa_subtree" in config else 3
        )
        min_taxa_base = (
            config["min_taxa_base"] if "min_taxa_base" in config else 7
        )

        for st in base_tree.traverse("postorder"):
            st_size = len(st)
            if (
                taxa_count - st_size >= min_taxa_base
                and st_size >= min_taxa_subtree
            ):
                subtrees.append(st)

        assert len(subtrees) > 0, "Failed to find any valid subtrees to prune"

        random_subtree_index = numpy.random.randint(0, len(subtrees))
        random_subtree = subtrees[random_subtree_index]
        subtree_leaves = set(n.name for n in random_subtree)
        keep_leaves = set(n.name for n in base_tree) - subtree_leaves

        base_tree.prune(
            keep_leaves,
            preserve_branch_length=True,
        )

        with open(output.tree, "w") as outfile:
            outfile.write(base_tree.write())

        with open(output.json, "w") as outfile:
            json.dump(
                {"pruned_leaves": list(subtree_leaves)},
                outfile,
            )


rule split_alignment:
    input:
        json="t_{tree_iter}/p_{pruning_iter}/pruning_info.json",
        alignment="t_{tree_iter}/align.fasta",
    output:
        reference_alignment_filename="t_{tree_iter}/p_{pruning_iter}/reference.fasta",
        query_alignment_filename="t_{tree_iter}/p_{pruning_iter}/query.fasta",
    run:
        base_alignment = utils.Alignment(input.alignment)
        with open(input.json) as infile:
            info_json = json.load(infile)

        (
            reference_alignment,
            query_alignment,
        ) = base_alignment.split_alignment(info_json["pruned_leaves"])

        with open(
            output.reference_alignment_filename,
            "w",
        ) as outfile:
            reference_alignment.write_fasta(outfile)

        with open(
            output.query_alignment_filename,
            "w",
        ) as outfile:
            query_alignment.write_fasta(outfile)


rule make_adna_damage_parameters:
    output:
        json="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/damage-params.json",
    run:
        model = config["exp_models"][int(wildcards.damage_iter)]
        params = models.make_adna_parameter_set(model)
        with open(output.json, "w") as outfile:
            json.dump(params.dict, outfile)


rule damage_query:
    input:
        align="t_{tree_iter}/p_{pruning_iter}/query.fasta",
        params_file="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/damage-params.json",
    output:
        damaged_align="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/damage.fasta",
    log:
        gzip="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/logs/pygargammel.log.gz",
        text="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/logs/pygargammel.log",
    run:
        model = config["exp_models"][int(wildcards.damage_iter)]

        params = models.make_adna_parameter_set(model)
        with open(input.params_file) as infile:
            params.load(json.load(infile))

        config_params = models.PyGargammelConfigParams()
        files = models.PyGargammelFiles(
            text=log.text,
            gzip=log.text,
            output=output.damaged_align,
            input=input.align,
        )

        pyg = models.PyGargammelConfig(
            pygargammel,
            params=params,
            config=config_params,
            files=files,
        )
        shell(pyg.command)


rule setup_aligners:
    input:
        reference="t_{tree_iter}/p_{pruning_iter}/reference.fasta",
        damage="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/damage.fasta",
    output:
        seqs="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/seqs.fasta",
    shell:
        "cat {input.reference} {input.damage} > {output.seqs}"


# Code for hmmer taken from PEWO


rule hmm_build:
    input:
        reference="t_{tree_iter}/p_{pruning_iter}/reference.fasta",
    output:
        hmm="t_{tree_iter}/p_{pruning_iter}/hmmer/profile.hmm",
    log:
        "t_{tree_iter}/p_{pruning_iter}/hmmer/profile.log",
    conda:
        "envs/hmmer.yaml"
    shell:
        "hmmbuild --cpu {threads} --dna {output.hmm} {input.reference} &> {log}"


rule align_hmmer:
    input:
        hmm="t_{tree_iter}/p_{pruning_iter}/hmmer/profile.hmm",
        reference="t_{tree_iter}/p_{pruning_iter}/reference.fasta",
        query="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/damage.fasta",
    output:
        psiblast="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/hmmer/align.psiblast",
    log:
        "t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/hmmer/align.log",
    conda:
        "envs/hmmer.yaml"
    shell:
        (
            "hmmalign "
            "--dna "
            "--outformat PSIBLAST "
            "-o {output.psiblast} "
            "--mapali {input.reference} "
            "{input.hmm} "
            "{input.query} "
            "&> {log}"
        )


rule hmmer_psiblast_to_fasta:
    input:
        psiblast="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/hmmer/align.psiblast",
    output:
        query="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/hmmer/align.fasta",
    run:
        with open(input.psiblast, "r") as f_in:
            lines = f_in.readlines()

            # dict to see which header has already been found
        headers = collections.OrderedDict()
        # when reading all seq ids at 1st block (before 1st empty line)
        # check if duplicate names
        firstblock = 1
        line_block = 0
        duplicate = {}  # map(line_block)=new_identifier
        duplicate_index = {}  # map(identifier)=#duplicate_envountered_in_block
        # read psiblast alignment
        for line in lines:
            # skip empty lines, reset block at empty lines
            if len(line.strip()) < 1:
                firstblock = 0
                line_block = 0
                continue
                # load sequences
            elts = line.strip().split()

            # identifier never encountered, register it
            if elts[0] not in headers:
                headers[elts[0]] = elts[1]
            else:
                # if still in block 1
                if (firstblock == 1) and (elts[0] in headers):
                    # set counter of how many times we encountered this id
                    if elts[0] not in duplicate_index:
                        duplicate_index[elts[0]] = 0
                    else:
                        duplicate_index[elts[0]] = duplicate_index[elts[0]] + 1
                        # create new id
                    duplicate[line_block] = (
                        elts[0] + "_" + str(duplicate_index[elts[0]])
                    )
                    print(
                        "duplicate at "
                        + str(line_block)
                        + " id set to "
                        + duplicate[line_block]
                    )
                    headers[duplicate[line_block]] = ""
                if line_block in duplicate:
                    headers[duplicate[line_block]] = (
                        headers[duplicate[line_block]] + elts[1]
                    )
                else:
                    headers[elts[0]] = headers[elts[0]] + elts[1]

            line_block = line_block + 1

        with open(output.query, "w") as f_out:
            for key in headers.keys():
                f_out.write(">" + key + "\n" + headers[key] + "\n")


rule align_muscle:
    input:
        seqs="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/seqs.fasta",
    output:
        align="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/muscle/align.fasta",
    conda:
        "envs/muscle.yaml"
    log:
        "t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/muscle/align.log",
    shell:
        "muscle -super5 {input.seqs} -output {output.align} &> {log}"


rule align_mafft:
    input:
        seqs="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/seqs.fasta",
    output:
        align="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/mafft/align.fasta",
    conda:
        "envs/mafft.yaml"
    shell:
        (
            "mafft "
            + "--quiet "
            + "--auto "
            + "{input.seqs} > "
            + "{output.align}"
        )


rule align_clustalo:
    input:
        seqs="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/seqs.fasta",
    output:
        align="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/clustalo/align.fasta",
    conda:
        "envs/clustalo.yaml"
    shell:
        "clustalo --in {input.seqs} --out {output.align}"


for tool in tools:

    rule:
        input:
            align=f"t_{{tree_iter}}/p_{{pruning_iter}}/d_{{damage_iter}}/{tool}/align.fasta",
            info=f"t_{{tree_iter}}/p_{{pruning_iter}}/pruning_info.json",
        output:
            reference=f"t_{{tree_iter}}/p_{{pruning_iter}}/d_{{damage_iter}}/{tool}/reference.fasta",
            query=f"t_{{tree_iter}}/p_{{pruning_iter}}/d_{{damage_iter}}/{tool}/query.fasta",
        run:
            base_alignment = utils.Alignment(input.align)
            with open(input.info) as infile:
                info_json = json.load(infile)

            (
                reference_alignment,
                query_alignment,
            ) = base_alignment.split_alignment(info_json["pruned_leaves"])

            with open(output.reference, "w") as outfile:
                reference_alignment.write_fasta(outfile)

            with open(output.query, "w") as outfile:
                query_alignment.write_fasta(outfile)



rule place_epang:
    input:
        tree="t_{tree_iter}/p_{pruning_iter}/pruned_tree.nwk",
        reference="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/{tool}/reference.fasta",
        query="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/{tool}/query.fasta",
    output:
        jplace="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/epa-ng/{tool}/epa_result.jplace",
        dir=directory(
            "t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/epa-ng/{tool}/"
        ),
    log:
        info="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/epa-ng/{tool}/epa_info.log",
        run="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/epa-ng/{tool}/epa_run.log",
    params:
        model=lookup(
            dpath="exp_trees/{tree_iter}/alignment/model", within=config
        ),
    conda:
        "envs/epang.yaml"
    shell:
        "epa-ng "
        "--tree {input.tree} "
        "--ref-msa {input.reference} "
        "--query {input.query} "
        "--model {params.model} "
        "--outdir {output.dir} "
        "--preserve-rooting on "
        "--no-pre-mask "
        "&> {log.run}"


rule compute_distances:
    input:
        jplace=expand(
            "t_{{tree_iter}}/p_{{pruning_iter}}/d_{{damage_iter}}/epa-ng/{tool}/epa_result.jplace",
            tool=tools,
        ),
        json="t_{tree_iter}/p_{pruning_iter}/pruning_info.json",
        tree="t_{tree_iter}/tree.nwk",
        params="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/damage-params.json",
    output:
        csv="t_{tree_iter}/p_{pruning_iter}/d_{damage_iter}/epa-ng/distances.csv",
    params:
        tools=tools,
    run:
        true_tree = ete3.Tree(open(input.tree).read())
        removed_taxa = set(json.load(open(input.json))["pruned_leaves"])

        model = config["exp_models"][int(wildcards.damage_iter)]
        adna_params = models.make_adna_parameter_set(model).dict

        with open(output.csv, "w") as outfile:
            csv_file = csv.DictWriter(
                outfile,
                csv_fields,
            )
            csv_file.writeheader()
            for tool, jp_file in zip(params.tools, input.jplace):
                jp = jplace.Jplace(json.load(open(jp_file)), true_tree)
                jp.set_true_tree(true_tree, removed_taxa)
                jp.compute_nds()

                for p in jp.placements:
                    csv_file.writerow(
                        p.json()
                        | {
                            "aligner": tool,
                            "tree_iter": wildcards.tree_iter,
                            "pruning_iter": wildcards.pruning_iter,
                            "damage_iter": wildcards.damage_iter,
                        }
                        | adna_params
                    )


rule coalece_distances:
    input:
        csvs=make_intermediat_results_list(),
    output:
        final_csv="distances.csv",
    run:
        with open(output.final_csv, "w") as csv_file:
            writer = csv.DictWriter(csv_file, csv_fields)
            writer.writeheader()

            for input_file in input.csvs:
                reader = csv.DictReader(open(input_file))
                for row in reader:
                    writer.writerow(row)
