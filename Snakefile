# Snakefile
import os

# Load parameters from the config file
configfile: "config/config.yaml"

# Shortcut variables
RAW_DIR = config["dirs"]["raw"]
NORM_DIR = config["dirs"]["normalized"]
PROC_DIR = config["dirs"]["processed"]
SCRIPTS_DIR = config["dirs"]["scripts"]
LOG_DIR = config["dirs"]["logs"]
CORPORA = config["corpora"]
SAMPLES = config["samples"]
MODEL_TYPES = config["model_types"]
PIVOT = config["pivot_sample"]

# Ensure directories exist
for d in [NORM_DIR, PROC_DIR, LOG_DIR, "results/vocab", "results/figures"]:
    os.makedirs(d,exist_ok=True)

# ------------------------------------------------------------------
# TARGETS
# ------------------------------------------------------------------
rule all:
    input:
        expand(os.path.join(PROC_DIR,"{sample}.txt"),sample=CORPORA.keys()),
        expand("results/vocab/{sample}.csv",sample=CORPORA.keys()),
        expand("results/figures/{sample}_dist.png",sample=CORPORA.keys()),
        "results/figures/combined_distribution.png",
        os.path.join(LOG_DIR,"preprocessing_summary.log"),
        expand("results/models/{sample}_w2v.model", sample=CORPORA.keys()),
        expand("results/models/{sample}_fasttext.model", sample=CORPORA.keys()),
        "results/metrics/anchor_risk_assessment.txt",
        "results/metrics/alignment_quality_table.txt",
        "results/figures/signatures/comparative_model_trend_annotated.png",
        "results/metrics/validation_stats.txt",
        expand("results/metrics/displacement_ranking_{mtype}.csv", mtype=MODEL_TYPES),
        expand("results/figures/comparative/displacement_scatter_{mtype}.png", mtype=MODEL_TYPES)

# ------------------------------------------------------------------
# RULES
# ------------------------------------------------------------------

rule lemmatize:
    """Step 1: JSON -> Normalized JSON"""
    input:
        corpus=os.path.join(RAW_DIR,"{sample}.json")
    output:
        norm=os.path.join(NORM_DIR,"{sample}_norm.json")
    params:
        script=lambda wildcards: os.path.join(SCRIPTS_DIR,f"preprocess_{CORPORA[wildcards.sample]}.py")
    log:
        os.path.join(LOG_DIR,"preprocess","{sample}_lemmatize.log")
    shell:
        "python {params.script} --file {input.corpus} --out_dir {NORM_DIR} > {log} 2>&1"

rule extract_to_txt:
    """Step 2: Normalized JSON -> Plain TXT"""
    input:
        norm_json=os.path.join(NORM_DIR,"{sample}_norm.json"),
        script=os.path.join(SCRIPTS_DIR,"extract_lemmas.py")
    output:
        txt=os.path.join(PROC_DIR,"{sample}.txt")
    log:
        os.path.join(LOG_DIR,"preprocess","{sample}_extract.log")
    shell:
        "python {input.script} --input {input.norm_json} --output {output.txt} > {log} 2>&1"

rule analyze_vocab:
    """Step 3a: Calculate Stats (Frequency + Entropy) -> CSV"""
    input:
        txt=os.path.join(PROC_DIR,"{sample}.txt"),
        script=os.path.join(SCRIPTS_DIR,"analyze_vocab.py")
    output:
        csv="results/vocab/{sample}.csv"
    log:
        "logs/preprocess/{sample}_analyze.log"
    shell:
        "python {input.script} --input {input.txt} --output {output.csv} >> logs/preprocessing_summary.log 2>&1"

rule plot_vocab:
    """
    Step 3b: Visualize Stats with Labels (Individual plots).
    """
    input:
        csv="results/vocab/{sample}.csv",
        script=os.path.join(SCRIPTS_DIR,"plot_vocab.py")
    output:
        plot="results/figures/{sample}_dist.png"
    log:
        "logs/preprocess/{sample}_plot.log"
    shell:
        """
        python {input.script} \
            --input {input.csv} \
            --output {output.plot} \
            --top_low 20 \
            --top_high 30 \
            >> logs/preprocessing_summary.log 2>&1
        """

rule plot_combined:
    """
    Step 3c: Generate combined comparison plot (1x3).
    """
    input:
        roman="results/vocab/roman.csv",
        napoleon="results/vocab/napoleon.csv",
        eu="results/vocab/eu.csv",
        script=os.path.join(SCRIPTS_DIR,"plot_vocab_combined.py")
    output:
        png="results/figures/combined_distribution.png"
    log:
        "logs/plot_combined.log"
    shell:
        """
        python {input.script} \
            --roman {input.roman} \
            --napoleon {input.napoleon} \
            --eu {input.eu} \
            --output {output.png} > {log} 2>&1
        """

rule train_models:
    """Step 4: Normalized JSON -> W2V & FastText Models"""
    input:
        norm_json=os.path.join(NORM_DIR, "{sample}_norm.json"),
        script=os.path.join(SCRIPTS_DIR, "train_models.py")
    output:
        w2v="results/models/{sample}_w2v.model",
        ft="results/models/{sample}_fasttext.model"
    log:
        os.path.join(LOG_DIR, "train", "{sample}_training.log")
    shell:
        """
        python {input.script} --input {input.norm_json} --output_prefix results/models/{wildcards.sample} --type w2v > {log} 2>&1
        python {input.script} --input {input.norm_json} --output_prefix results/models/{wildcards.sample} --type fasttext >> {log} 2>&1
        """

rule assess_anchor_stability:
    """
    Step: Semantic Anchor Stability Assessment.

    Generates a LaTeX-formatted table containing stability metrics 
    (Second-Order cosine similarity) for each triplet across W2V and FastText models.

    Execution: Runs AFTER individual model training but BEFORE alignment (Procrustes).
    """
    input:
        script="scripts/assess_anchor_stability.py",
        anchors="data/reference/alignment_anchors.json",
        # CSV files are required to display term frequencies (n) in the table
        vocab=expand("results/vocab/{sample}.csv",sample=["roman", "napoleon", "eu"]),
        # Trained models are required to calculate vector similarities
        w2v=expand("results/models/{sample}_w2v.model",sample=["roman", "napoleon", "eu"]),
        ft=expand("results/models/{sample}_fasttext.model",sample=["roman", "napoleon", "eu"])
    output:
        # The output is saved as a text file containing the LaTeX table rows
        report="results/metrics/anchor_risk_assessment.txt"
    params:
        models_dir="results/models",
        vocab_dir="results/vocab"
    shell:
        """
        mkdir -p results/metrics

        python {input.script} \
            --anchors {input.anchors} \
            --models_dir {params.models_dir} \
            --vocab_dir {params.vocab_dir} > {output.report}
        """

rule align_all:
    """
    Main goal: Align all models to the pivot defined in config.
    """
    input:
        expand("results/models/{sample}_aligned_{mtype}.model", sample=SAMPLES, mtype=MODEL_TYPES)

rule align_to_pivot:
    """
    Align non-pivot models to the pivot using Orthogonal Procrustes.
    """
    input:
        src = "results/models/{sample}_{mtype}.model",
        tgt = f"results/models/{PIVOT}_{{mtype}}.model",
        anchors = config["anchors_file"],
        script = "scripts/align_models.py"
    output:
        "results/models/{sample}_aligned_{mtype}.model"
    wildcard_constraints:
        # Dynamically exclude pivot from source samples
        sample = "|".join([s for s in SAMPLES if s != PIVOT])
    params:
        # Determine source language key for the alignment script
        src_lang = lambda wildcards: "fr" if wildcards.sample == "napoleon" else "en",
        tgt_lang = "la"
    shell:
        "python {input.script} --source {input.src} --target {input.tgt} "
        "--anchors {input.anchors} --output {output} "
        "--source_lang {params.src_lang} --target_lang {params.tgt_lang}"

rule pivot_copy:
    """
    Create aligned naming convention for the pivot sample.
    """
    input:
        f"results/models/{PIVOT}_{{mtype}}.model"
    output:
        f"results/models/{PIVOT}_aligned_{{mtype}}.model"
    shell:
        "cp {input} {output} && "
        "if [ -f {input}.wv.vectors.npy ]; then cp {input}.wv.vectors.npy {output}.wv.vectors.npy; fi; "
        "if [ -f {input}.wv.vectors_ngrams.npy ]; then cp {input}.wv.vectors_ngrams.npy {output}.wv.vectors_ngrams.npy; fi"


rule check_alignment_quality:
    """
    Step: Procrustes Alignment Quality Assessment (Table 2).
    Compares cosine similarities between anchor terms in models "Before" and "After" alignment.
    """
    input:
        script="scripts/assess_procrustes_quality.py",
        # Use the specific anchor triplets file intended for alignment
        anchors="data/reference/alignment_anchors.json",

        # Wait for ALL models to be ready (both raw and aligned versions)
        # SAMPLES are defined in config (roman, napoleon, eu)
        models_raw=expand("results/models/{sample}_{mtype}.model",
            sample=SAMPLES,mtype=MODEL_TYPES),
        models_aligned=expand("results/models/{sample}_aligned_{mtype}.model",
            sample=SAMPLES,mtype=MODEL_TYPES)
    output:
        # Result: text file containing the formatted table
        report="results/metrics/alignment_quality_table.txt"
    params:
        models_dir="results/models"
    shell:
        """
        python {input.script} \
            --anchors {input.anchors} \
            --models_dir {params.models_dir} > {output.report}
        """

rule analyze_technicality:
    """
    Step 6: Technicality Evolution Analysis (Density, Erosion, Validation).
    Generates heatmap, slope chart, and comparative cloud.
    """
    input:
        # Collect all aligned models
        models = expand("results/models/{sample}_aligned_{mtype}.model",
                        sample=SAMPLES, mtype=MODEL_TYPES),
        # Load JSON with terms and color highlights
        anchors = "data/reference/legal_anchors.json",
        # The analysis script
        script = "scripts/technicality_analysis.py"
    output:
        # Main publication-ready plot
        plot = "results/figures/signatures/comparative_model_trend_annotated.png",
        # Validation statistics
        stats = "results/metrics/validation_stats.txt",
        # (Optional) CSV files for verification
        csv_w2v = "results/metrics/technicality_w2v.csv",
        csv_ft = "results/metrics/technicality_fasttext.csv"
    shell:
        """
        # Create output directories on the fly
        mkdir -p results/metrics
        mkdir -p results/figures/signatures

        # Run the analysis
        python {input.script} \
            --models_dir results/models \
            --metrics_dir results/metrics \
            --fig_dir results/figures/signatures \
            --anchors {input.anchors}
        """

rule analyze_displacement:
    """
    Step 7: Semantic Displacement Analysis (All Models).
    """
    input:
        models=expand("results/models/{sample}_aligned_{mtype}.model",
            sample=SAMPLES,mtype=MODEL_TYPES),
        anchors="data/reference/legal_anchors.json",
        script="scripts/semantic_displacement_analysis.py"
    output:
        expand("results/metrics/displacement_ranking_{mtype}.csv",mtype=MODEL_TYPES),
        expand("results/figures/comparative/displacement_bar_{mtype}.png",mtype=MODEL_TYPES),
        expand("results/figures/comparative/displacement_scatter_{mtype}.png",mtype=MODEL_TYPES)
    shell:
        """
        mkdir -p results/metrics
        mkdir -p results/figures/comparative

        python {input.script} \
            --models_dir results/models \
            --metrics_dir results/metrics \
            --fig_dir results/figures/comparative \
            --anchors {input.anchors}
        """


rule finalize_logs:
    """
    Helper rule to collect logs into summary.
    """
    input:
        expand(os.path.join(LOG_DIR,"preprocess","{sample}_{step}.log"),
            sample=CORPORA.keys(),step=["lemmatize", "extract"])
    output:
        summary=os.path.join(LOG_DIR,"preprocessing_summary.log")
    shell:
        "cat {input} > {output}"
