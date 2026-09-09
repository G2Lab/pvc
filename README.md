# PVC

**Privacy-preserving variant calling.**

This repository contains the core research implementation.

## How it works

1. Prepare public graph, panel, k-mer, and haplotype-block structures.
2. Compute evidence from the client's read counts and coverage.
3. Score candidate haplotype pairs using plaintext computation or three-party
   secure computation.
4. Reconstruct the private protocol's outputs at the client and produce genotype
   calls.

| Protocol | Private computation | Client receives |
| --- | --- | --- |
| Light | Scores secret-shared, client-computed emissions | Candidate score vectors |
| Medium | Scores secret-shared emissions and selects the winner privately | Winning one-hot vector |
| Heavy | Computes emissions from shared count/coverage selectors, scores candidates, and selects the winner privately | Winning one-hot vector |

Public panel structure, LD blocks, and candidate haplotypes remain public.

## Code map

| Path | Purpose |
| --- | --- |
| `src/pvc/pangenome/` | Graphs, variants, k-mers, and probability tables |
| `src/pvc/index/` | Index preparation, LD blocks, and genotyping plans |
| `src/pvc/client/` | Client-side k-mer processing and reconstruction helpers |
| `src/pvc/genotype/` | Plaintext genotyping and VCF output |
| `src/pvc/genotype/private/` | Light, Medium, Heavy, role separation, and profiling |
| `src/crypto/CrypTen/` | Original upstream CrypTen runtime, configuration, and license |
| `src/pangenie/`, `src/gatk/` | PanGenie and GATK integration code |
| `tools/GenotypeConcordance/` | Genotype comparison utility |
| `tests/` | Secret-sharing and role-separated protocol checks |

## Runtime requirements

The tested runtime uses **Python 3.10, PyTorch 1.13.1, and NumPy < 2**.
The original [Facebook Research CrypTen](https://github.com/facebookresearch/CrypTen)
is included at commit `775868a02d6dac50774ce376a55b01fbd8bd85b6`.

## Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pvc --help
python -m pytest
```

## Run

```bash
pvc --mode heavy \
  --cereal /path/to/read_UniqueKmersMap.json \
  --graph /path/to/Graph.json \
  --panel-vcf /path/to/panel.vcf.gz \
  --blocks-file /path/to/panel.blocks.det \
  --mean-kmer-abundance 30 \
  --sample-name SAMPLE \
  --output results/calls.vcf
```

Choose `plaintext`, `light`, `medium`, or `heavy` with `--mode`; `python -m pvc`
provides the same interface. Private modes run on CPU with three computing-party
processes and a separate upstream trusted preprocessing server. The parent
remains the client. `PVC_THREADS_PER_ROLE` controls PyTorch worker threads
(default `1`); `PVC_MPC_TIMEOUT_SECONDS` controls the local timeout (default
`3600`).

PLINK and bcftools are needed if `--blocks-file` is omitted. PanGenie, GATK,
and Jellyfish are needed only for preprocessing/comparison workflows that use
them. Native executables and reference datasets are supplied separately.

## Run a chromosome comparison with Slurm

Use the same prepared chromosome inputs for all four modes. For chr21, supply
its graph, read-count JSON, panel, blocks, and sample truth VCF. Install PVC in
`.venv` as above, then save this as `validate.sbatch` in the repository root and
replace the input paths:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pvc-chr21-check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=results/chr21/logs/%A_%a.out
#SBATCH --error=results/chr21/logs/%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

modes=(plaintext light medium heavy)
mode="${modes[$SLURM_ARRAY_TASK_ID]}"
out="results/chr21/$SLURM_ARRAY_JOB_ID/$mode"
mkdir -p "$out"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PVC_THREADS_PER_ROLE=1
export PVC_MPC_TIMEOUT_SECONDS=42600 PVC_OT_PAIRWISE_PRF_PADS=1
export CUDA_VISIBLE_DEVICES=""

.venv/bin/python -u -m pvc --mode "$mode" \
  --cereal /path/to/chr21/read_UniqueKmersMap.json \
  --graph /path/to/chr21/Graph.json \
  --panel-vcf /path/to/chr21/panel.vcf.gz \
  --blocks-file /path/to/chr21/plink_blocks.blocks.det \
  --mean-kmer-abundance 30 \
  --sample-name HG00438 \
  --vcf /path/to/chr21/truth.vcf.gz \
  --output "$out/calls.vcf" \
  > "$out/stdout.log" 2> "$out/stderr.log"
```

Submit from the repository root. Create the Slurm log directory before submitting:

```bash
mkdir -p results/chr21/logs
job_id=$(sbatch --parsable validate.sbatch)
job_id=${job_id%%;*}
squeue -j "$job_id"
sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

The array runs plaintext, Light, Medium, and Heavy, with at most two modes running
concurrently. Adjust the partition, memory, and time limit for your cluster;
128 GB and 12 hours are resource requests, not measured requirements. Each job
writes to a new directory under `results/chr21/<job_id>/`. This reruns genotyping
from prepared inputs; it does not rebuild the graph or process raw reads.