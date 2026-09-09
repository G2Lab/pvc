
def read_vcf_file(vcf_file):
    genotypes = {}
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            pos = int(parts[1])

            genotype_info = parts[9]
            genotype_str = genotype_info.split(':')[0]
            if genotype_str == '0/0':
                genotype = (0, 0)
            elif genotype_str == '0/1':
                genotype = (0, 1)
            elif genotype_str == '1/1':
                genotype = (1, 1)
            else:
                genotype = (-1, -1)
            genotypes[pos] = genotype
    return genotypes

def verify_genotype_outputs(genotype_results_arr, Graph, vcf_file):
    assert len(genotype_results_arr) == len(Graph.variants), "Mismatch in number of variants between genotype results and graph."

    true_genotypes = read_vcf_file(vcf_file)    

    correct_genotypes = 0
    total_variants = 0

    incorrect_variants = []

    for i in range(len(genotype_results_arr)):
        try:
            variant = Graph.variants[i]
            if variant.is_combined():
                singleton_variants, singleton_genotype_results = variant.separate_variants(genotype_results_arr[i], True)
            else:
                singleton_variants = [variant]
                singleton_genotype_results = [genotype_results_arr[i]]

            for singleton_variant, singleton_genotype_result in zip(singleton_variants, singleton_genotype_results):
                variant_pos = singleton_variant.start_position + 1


                if variant_pos not in true_genotypes:
                    print(f"Warning: Variant position {variant_pos} not found in VCF genotypes.")
                    continue
                
                true_genotype = true_genotypes[variant_pos]

                if true_genotype == (-1, -1):
                    continue
                
                total_variants += 1
                
                # pseudo-code, assuming you have is_undefined_allele and nr_of_alleles
                nr_alleles = singleton_variant.nr_of_alleles()
                defined_alleles = [0]
                for a in range(1, nr_alleles):
                    if not singleton_variant.is_undefined_allele(a):
                        defined_alleles.append(a)

                if singleton_genotype_result.contains_no_likelihoods():
                    singleton_genotype_result.add_to_likelihood(0, 0, 1.0)

                if len(defined_alleles) < nr_alleles:
                    singleton_genotype_result = singleton_genotype_result.get_specific_likelihoods(defined_alleles)

                most_likely_genotype = singleton_genotype_result.get_likeliest_genotype()

                if most_likely_genotype == true_genotype or most_likely_genotype == (true_genotype[1], true_genotype[0]):
                    correct_genotypes += 1
                else:
                    incorrect_variants.append(
                        f"Variant at position {variant_pos}: True genotype {true_genotype}, Predicted genotype {most_likely_genotype}"
                    )
        except:
            continue
    
    accuracy = correct_genotypes / total_variants if total_variants > 0 else 0.0
    print(f"Genotype verification completed. Accuracy: {accuracy:.2%} ({correct_genotypes}/{total_variants})")

    if incorrect_variants:
        # Summarize errors by type
        error_types = {}
        for msg in incorrect_variants:
            # Extract true and predicted genotypes
            if "True genotype" in msg and "Predicted genotype" in msg:
                true_gt = msg.split("True genotype ")[1].split(",")[0].replace("(", "") + "," + msg.split("True genotype ")[1].split(", ")[1].split(")")[0]
                pred_gt = msg.split("Predicted genotype ")[1].replace("(", "").replace(")", "").strip()
                key = f"({true_gt}) -> ({pred_gt})"
                error_types[key] = error_types.get(key, 0) + 1

        print(f"\nError summary ({len(incorrect_variants)} incorrect calls):")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {error_type}: {count}")
        if len(error_types) > 10:
            print(f"  ... and {len(error_types) - 10} more error types")

    return accuracy, correct_genotypes, total_variants
