import argparse

INDIVIDUALS = ["HG00138", "HG00635", "HG01112", "HG01600", "HG02698", "NA12778", "NA18853", "HG002_chr20_10k"]
TEST_DATASETS = ["tiny", "small", "medium", "chr21"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the privacy-preserving genotyping protocol for an individual."
    )

    parser.add_argument(
        "individual",
        choices=INDIVIDUALS + TEST_DATASETS,
        help=f"Individual ID to process. Choices: {INDIVIDUALS}"
    )

    parser.add_argument(
        "--mean-kmer-abundance",
        type=int,
        default=None,
        help="Mean k-mer abundance (auto-detected from histogram if not provided)"
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the genotyping protocol from scratch"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()
    return args.individual, args.restart, args.mean_kmer_abundance, args.verbose
