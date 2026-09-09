
import sys
import bisect
import os
from collections import defaultdict
from collections import namedtuple
import argparse
import math
from decimal import Decimal
import vcf
from enum import Enum

Position = namedtuple("Position", "chrom position")


def parse_bed(bed_path):
	"""Parse a BED file into sorted interval lists per chromosome."""
	regions = {}
	with open(bed_path) as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#") or line.startswith("track"):
				continue
			parts = line.split("\t")
			chrom = parts[0]
			start = int(parts[1])
			end = int(parts[2])
			if chrom not in regions:
				regions[chrom] = ([], [])
			regions[chrom][0].append(start)
			regions[chrom][1].append(end)
	for chrom in regions:
		starts, ends = regions[chrom]
		pairs = sorted(zip(starts, ends))
		regions[chrom] = ([s for s, _ in pairs], [e for _, e in pairs])
	total = sum(len(s) for s, _ in regions.values())
	sys.stderr.write('Loaded ' + str(total) + ' BED regions across ' + str(len(regions)) + ' chromosomes.\n')
	return regions


def pos_in_bed(regions, chrom, pos):
	"""Check if a 1-based position falls within any BED interval (0-based half-open)."""
	if chrom not in regions:
		return False
	pos0 = pos - 1
	starts, ends = regions[chrom]
	idx = bisect.bisect_right(starts, pos0) - 1
	if idx < 0:
		return False
	return pos0 < ends[idx]

class VariantType(Enum):
	snp = 0
	small_insertion = 1
	small_deletion = 2
	small_complex = 3
	midsize_insertion = 4
	midsize_deletion = 5
	midsize_complex = 6
	large_insertion = 7
	large_deletion = 8
	large_complex = 9

class Variant:
	def __init__(self, genotype, binary_genotype, variant_type : VariantType, is_nonref, quality=None, allele_frequency=None, unique_kmers=None, missing_alleles=None, alleles=None):
		"""
		Represents a genotyped variant.
		"""
		self._genotype=genotype
		self._binary_genotype=binary_genotype
		self._variant_type=variant_type
		self._quality=quality
		self._allele_frequency=allele_frequency
		self._unique_kmers=unique_kmers
		self._missing_alleles=missing_alleles
		self._alleles=alleles
		self.nonref = is_nonref # true if the genotype is different from 0/0, i.e. the sample carries at least one non-reference allele

	def get_genotype(self):
		return self._genotype

	def get_binary_genotype(self):
		return self._binary_genotype

	def is_type(self, vartype : VariantType):
		if vartype == self._variant_type:
			return True
		else:
			return False

	def get_type(self):
		return self._variant_type

	def get_quality(self):
		return self._quality

	def get_allele_frequency(self):
		return self._allele_frequency

	def get_unique_kmers(self):
		return self._unique_kmers

	def get_missing_alleles(self):
		return self._missing_alleles

	def get_alleles(self):
		return self._alleles

def determine_type_from_ids(ids):
	vartypes = [i.split('-')[2] for i in ids]
	if all([v == 'SNV' for v in vartypes]):
		return VariantType.snp

	# determine variant length
	allele_lengths = []
	for var_id in ids:
		if 'SNV' in var_id:
			continue
		assert var_id.split('-')[-2] in ['INS', 'DEL']
		length = int(var_id.split('-')[-1])
		allele_lengths.append(length)
	varlen = max(allele_lengths)

	is_insertion = 'INS' in vartypes and not 'DEL' in vartypes
	is_deletion = 'DEL' in vartypes and not 'INS' in vartypes

	if varlen < 20:
		if is_insertion:
			return VariantType.small_insertion
		if is_deletion:
			return VariantType.small_deletion
		return VariantType.small_complex

	if varlen >= 20 and varlen < 50:
		if is_insertion:
			return VariantType.midsize_insertion
		if is_deletion:
			return VariantType.midsize_deletion
		return VariantType.midsize_complex

	if varlen >= 50:
		if is_insertion:
			return VariantType.large_insertion
		if is_deletion:
			return VariantType.large_deletion
		return VariantType.large_complex
	assert(False)

def determine_type_from_record(record):
	"""
	Determines the variant type.
	"""
	ref = str(record.REF)
	alts = [str(alt) for alt in record.ALT]

	if record.is_snp:
		return VariantType.snp

	# Size an indel by its event length, not by the length of the VCF allele
	# strings (which include the anchoring base). This makes the bins correspond
	# to 1--19 bp, 20--49 bp, and >=50 bp events.
	length_deltas = [len(alt) - len(ref) for alt in alts]
	varlen = max(abs(delta) for delta in length_deltas)
	is_deletion = all(delta < 0 for delta in length_deltas)
	is_insertion = all(delta > 0 for delta in length_deltas)

	if varlen < 20:
		if is_insertion:
			return VariantType.small_insertion
		if is_deletion:
			return VariantType.small_deletion
		return VariantType.small_complex

	if varlen >= 20 and varlen < 50:
		if is_insertion:
			return VariantType.midsize_insertion
		if is_deletion:
			return VariantType.midsize_deletion
		return VariantType.midsize_complex

	if varlen >= 50:
		if is_insertion:
			return VariantType.large_insertion
		if is_deletion:
			return VariantType.large_deletion
		return VariantType.large_complex
	assert(False)


def extract_call(record, samples, read_gl=False, read_qual=False):
	"""
	Extract genotype information from a VCF Record.
	"""
	
	# list containing Variant for each sample (in order of samples)
	result = [None] * len(samples)

	# allele frequency of least frequent allele
	allele_frequency = 0.0
	unique_kmers = 0
	missing_alleles = 0
	
	# determine the allele frequency (of the least frequent genotype allele)
	if 'AF' in record.INFO:
		info_fields = record.INFO['AF']
		# add frequency of reference allele
		frequencies = [1.0 - sum(info_fields)] + info_fields
		allele_frequency = max(min(frequencies),0.0)

	# determine minimum number of kmers that cover each allele
	if 'AK' in record.INFO:
		info_fields = record.INFO['AK']
		# determine minimum of unique kmers (exclude alleles not covered by paths)
		unique_kmers = min([i for i in info_fields if i >= 0])

	# determine number of missing alleles at variant position
	if 'MA' in record.INFO:
		missing_alleles = record.INFO['MA']
	
	# determine the type of variant
	variant_type = determine_type_from_record(record)
	
	variant_alleles = tuple([str(record.REF)] + [str(alt) for alt in record.ALT])
	for call in record.samples:
		# genotype alleles
		genotype_sequences = None
		# binary genotype
		binary_genotype = 3
		# genotype quality
		likelihood = 0
		is_nonref = True
		sample_name = call.sample
		if sample_name not in samples:
			continue
		genotype_string = call['GT']
		if genotype_string is not None:
			genotype_string = genotype_string.replace('/', '|')
		genotype_list = genotype_string.split('|') if genotype_string else []
		if len(genotype_list) == 2 and all(allele != '.' for allele in genotype_list):
			genotype = (int(genotype_list[0]), int(genotype_list[1]))
			if all(0 <= allele < len(variant_alleles) for allele in genotype):
				is_nonref = genotype[0] != 0 or genotype[1] != 0
				# Retain allele dosage while treating 0/1 and 1/0 as the same
				# unphased genotype.
				genotype_sequences = tuple(sorted([
					variant_alleles[genotype[0]],
					variant_alleles[genotype[1]],
				]))

				# compute likelihood of genotype
				if read_gl:
					# check if GQ field is present
					if 'GQ' in record.FORMAT.split(':') and call['GQ'] is not None:
						likelihood = int(call['GQ'])
				if read_qual:
					if record.QUAL is not None:
						likelihood = int(record.QUAL)

				# Store dosage only for a biallelic record. A multiallelic
				# call is mapped to the truth alleles during evaluation.
				if len(variant_alleles) == 2:
					binary_genotype = genotype[0] + genotype[1]
				else:
					binary_genotype = -1

		result[samples.index(sample_name)] = Variant(
			genotype_sequences,
			binary_genotype,
			variant_type,
			is_nonref,
			likelihood,
			allele_frequency,
			unique_kmers,
			missing_alleles,
			variant_alleles,
		)

	return Position(record.CHROM, record.POS), result


def map_call_to_truth_biallelic(truth_gt, call_gt):
	"""Map a call genotype to a fixed biallelic truth allele space.

	Returns columns 0, 1, or 2 for exact REF/ALT dosage, 4 for an absent or
	missing genotype, and 3 for a typed genotype containing an allele sequence
	that is not the truth REF or ALT. Multiallelic call records are therefore
	evaluated rather than silently dropped.
	"""
	if call_gt is None or call_gt.get_genotype() is None:
		return 4
	truth_alleles = truth_gt.get_alleles()
	if truth_alleles is None or len(truth_alleles) != 2:
		raise ValueError("Truth genotype is not biallelic.")
	truth_ref, truth_alt = truth_alleles
	call_sequences = call_gt.get_genotype()
	if len(call_sequences) != 2:
		return 3
	if any(allele not in (truth_ref, truth_alt) for allele in call_sequences):
		return 3
	return sum(allele == truth_alt for allele in call_sequences)


def allele_presence_counts(truth_gt, call_gt):
	"""Return sequence-aware non-reference allele TP, FP, and FN counts."""
	truth_ref, truth_alt = truth_gt.get_alleles()
	truth_nonref = {truth_alt} if truth_gt.get_binary_genotype() in (1, 2) else set()
	called_nonref = (
		set(call_gt.get_genotype()) - {truth_ref}
		if call_gt is not None and call_gt.get_genotype() is not None
		else set()
	)
	return (
		len(truth_nonref & called_nonref),
		len(called_nonref - truth_nonref),
		len(truth_nonref - called_nonref),
	)


class GenotypingStatistics:

	def __init__(self, quality, allele_frequency, unique_kmers, missing_alleles):

		# thresholds used to filter variant set
		self.quality = quality
		self.allele_frequency = allele_frequency
		self.unique_kmers = unique_kmers
		self.missing_alleles = missing_alleles

		# numbers of variants in baseline and callset
		self.total_baseline = 0
		self.total_baseline_nonref = 0
		self.total_baseline_biallelic = 0
		self.total_intersection = 0

		# fractions of correct, wrong and untyped variants
		self.correct_all = 0
		self.correct_nonref = 0
		self.wrong_all = 0
		self.wrong_nonref = 0
		self.not_typed_all = 0
		self.not_typed_nonref = 0

		self.not_in_callset_all = 0
		self.not_in_callset_biallelic = 0
		self.not_in_callset_nonref = 0
		self.absent_truth = 0
		self.allele_presence_tp = 0
		self.allele_presence_fp = 0
		self.allele_presence_fn = 0

		# Fixed-denominator, sequence-aware biallelic confusion matrix.
		# Truth rows: 0/0, 0/1, 1/1.
		# Call columns: 0/0, 0/1, 1/1, wrong/unknown allele, missing.
		self.confusion_matrix = [[0 for x in range(5)] for y in range(3)]

	def print_stats_to_file(self, tsv_file):
		typed_all = max(self.correct_all + self.wrong_all, 1)
		typed_nonref = max(self.correct_nonref + self.wrong_nonref, 1)
		assert self.total_baseline == self.correct_all + self.wrong_all + self.not_in_callset_all + self.not_typed_all
		assert self.total_baseline_nonref == self.correct_nonref + self.wrong_nonref + self.not_in_callset_nonref + self.not_typed_nonref
		correct_biallelic = sum(self.confusion_matrix[row][row] for row in range(3))
		wrong_biallelic = sum(
			self.confusion_matrix[row][col]
			for row in range(3)
			for col in (0, 1, 2, 3)
			if col != row
		)
		not_typed_biallelic = sum(self.confusion_matrix[row][4] for row in range(3))
		assert self.total_baseline_biallelic == correct_biallelic + wrong_biallelic + not_typed_biallelic
		typed_biallelic = max(correct_biallelic + wrong_biallelic, 1)

		tsv_file.write('\t'.join([
			str(self.quality), # quality
			str(self.allele_frequency), # allele freq
			str(self.unique_kmers), # unique kmer count
			str(self.missing_alleles), # nr of missing alleles
			str(self.total_baseline), # total baseline variants
			str(self.total_baseline_biallelic), # total biallelic variants
			str(self.total_baseline_nonref), # total nonref variants
			str(self.total_intersection), # intersection of callsets
			str(self.correct_all/float(typed_all)), # correct
			str(self.wrong_all/float(typed_all)), # wrong
			str((self.not_typed_all+self.not_in_callset_all)/float(self.total_baseline if self.total_baseline != 0 else 1)), # not typed
			str(correct_biallelic/ float(typed_biallelic)), # correct biallelic
			str(wrong_biallelic/ float(typed_biallelic)), # wrong biallelic
			str(not_typed_biallelic/float(self.total_baseline_biallelic if self.total_baseline_biallelic != 0 else 1)),  # not typed biallelic
			str(self.correct_nonref/float(typed_nonref)), # correct nonref
			str(self.wrong_nonref/float(typed_nonref)), # wrong nonref
			str((self.not_typed_nonref+self.not_in_callset_nonref)/float(self.total_baseline_nonref if self.total_baseline_nonref != 0 else 1)),  # not typed nonref
			str(self.correct_all),
			str(self.wrong_all),
			str(self.not_typed_all),
			str(self.not_in_callset_all),
			str(correct_biallelic),
			str(wrong_biallelic),
			str(not_typed_biallelic),
			str(self.not_in_callset_biallelic),
			str(self.correct_nonref),
			str(self.wrong_nonref),
			str(self.not_typed_nonref),
			str(self.not_in_callset_nonref),
			str(self.allele_presence_tp),
			str(self.allele_presence_fp),
			str(self.allele_presence_fn),
			]) + '\n'
											)

	def print_matrix_to_file(self, txt_file):
		txt_file.write('\n###############################################################################################################################################\n')
		txt_file.write('#       Matrix for quality: '  + str(self.quality) +', allele frequency threshold: ' + str(self.allele_frequency) + ' , unique kmer count threshold: ' + str(self.unique_kmers) + ' and missing alleles threshold: ' + str(self.missing_alleles) + '#\n')
		txt_file.write('###############################################################################################################################################\n\n')
		txt_file.write('# matrix_schema=sequence-aware-v2\n')
		matrix_string = "\t0\t1\t2\tOTHER\t./.\n"
		for true in range(3):
			line = str(true) + '\t'
			for geno in range(5):
				line += str(self.confusion_matrix[true][geno]) + '\t'
			matrix_string += line + '\n'
		txt_file.write(matrix_string)


class GenotypeConcordanceComputer:
	def __init__(self, baseline_vcf, callset_vcf, samples, use_qual):
		self._baseline_variants = {}
		self._callset_variants = defaultdict(list)
		self._duplicated_positions = defaultdict(lambda:False)
		self._total_baseline = 0
		self._total_callset = 0
		self._samples = samples
		self._quiet_duplicate_warnings = os.environ.get('PVC_CONCORDANCE_QUIET_DUPLICATES', '0') == '1'
		self._duplicate_baseline_records = 0

		# read baseline variants
		for record in vcf.Reader(filename=baseline_vcf):
			# determine variant type
			pos, variants = extract_call(record, self._samples)
			if pos in self._baseline_variants:
				# duplicated position, ignore it
				self._duplicate_baseline_records += 1
				if not self._quiet_duplicate_warnings:
					sys.stderr.write('Warning: position ' + str(pos.chrom) + ' ' + str(pos.position) + ' occurs more than once and will be skipped.\n')
				self._duplicated_positions[pos] = True
				continue
			self._total_baseline += 1
			self._baseline_variants[pos] = variants

		# read callset variants
		for record in vcf.Reader(filename=callset_vcf):
			if use_qual:
				pos, variants = extract_call(record, self._samples, read_qual = True)
			else:
				pos, variants = extract_call(record, self._samples, read_gl = True)
			self._callset_variants[pos].append(variants)
			self._total_callset += 1

		sys.stderr.write('Read ' + str(self._total_baseline) + ' variants from the baseline VCF (including duplicates).\n')
		sys.stderr.write('Read ' + str(self._total_callset) + ' variants from the callset VCF (including duplicates).\n')
		if self._quiet_duplicate_warnings and self._duplicate_baseline_records:
			sys.stderr.write('Skipped ' + str(self._duplicate_baseline_records) + ' duplicate baseline records.\n')

	def nr_baseline_variants(self):
		return self._total_baseline

	def nr_callset_variants(self):
		return self._total_callset

	def print_statistics(self, sample, varianttypes, quality, allele_freq, uk_count, missing_count, txt_files, tsv_files, bed_regions=None):
		"""
		Compute genotype concordance statistics for all variants with
		a quality at least quality and for which the allele frequency
		of the least frequent allele is at least allele_freq.
		If bed_regions is provided, only variants within those regions are evaluated.
		"""

		# make sure there is a txt and tsv file for each variant type
		assert len(txt_files) == len(tsv_files) == len(varianttypes)

		# keep variant statistics for all variant types
		statistics = { var_type : GenotypingStatistics(quality, allele_freq, uk_count, missing_count) for var_type in varianttypes }
		multiple_prediction_positions = 0
		sample_id = self._samples.index(sample)
		# check genotype predictions
		for pos, genotypes in self._baseline_variants.items():

			gt = genotypes[sample_id]
			# if position was present multiple times in baseline, skip it
			if self._duplicated_positions[pos]:
				continue

			# if BED regions provided, skip variants outside them
			if bed_regions is not None and not pos_in_bed(bed_regions, pos.chrom, pos.position):
				continue

			# if true genotype is unknown, skip
			if gt is None or gt.get_genotype() == None:
				continue
			vartype = gt.get_type()

			statistics[vartype].total_baseline += 1

			if gt.nonref:
				statistics[vartype].total_baseline_nonref += 1

			callset_predictions = self._callset_variants[pos]

			truth_is_biallelic = gt.get_binary_genotype() in (0, 1, 2) and len(gt.get_alleles()) == 2
			if truth_is_biallelic:
				statistics[vartype].total_baseline_biallelic += 1

			if len(callset_predictions) > 1:
				multiple_prediction_positions += 1
				if not self._quiet_duplicate_warnings:
					sys.stderr.write(
						'Warning: multiple predictions for variant at position '
						+ str(pos.chrom) + ' ' + str(pos.position)
						+ '; treating the call as untyped.\n'
					)

			if len(callset_predictions) > 0:
				statistics[vartype].total_intersection += 1
				callset_gt = (
					callset_predictions[0][sample_id]
					if len(callset_predictions) == 1
					else None
				)

				if callset_gt is None or callset_gt.get_genotype() is None:
					statistics[vartype].not_typed_all += 1
					if gt.nonref:
						statistics[vartype].not_typed_nonref += 1
					if truth_is_biallelic:
						statistics[vartype].confusion_matrix[gt.get_binary_genotype()][4] += 1
						tp, fp, fn = allele_presence_counts(gt, None)
						statistics[vartype].allele_presence_tp += tp
						statistics[vartype].allele_presence_fp += fp
						statistics[vartype].allele_presence_fn += fn
					continue

				if callset_gt.get_quality() < quality or callset_gt.get_allele_frequency() < allele_freq or callset_gt.get_unique_kmers() < uk_count or callset_gt.get_missing_alleles() < missing_count:
					statistics[vartype].not_typed_all += 1
					if gt.nonref:
						statistics[vartype].not_typed_nonref += 1
					if truth_is_biallelic:
						statistics[vartype].confusion_matrix[gt.get_binary_genotype()][4] += 1
						tp, fp, fn = allele_presence_counts(gt, None)
						statistics[vartype].allele_presence_tp += tp
						statistics[vartype].allele_presence_fp += fp
						statistics[vartype].allele_presence_fn += fn
					continue

				if truth_is_biallelic:
					call_column = map_call_to_truth_biallelic(gt, callset_gt)
					statistics[vartype].confusion_matrix[gt.get_binary_genotype()][call_column] += 1
					tp, fp, fn = allele_presence_counts(gt, callset_gt)
					statistics[vartype].allele_presence_tp += tp
					statistics[vartype].allele_presence_fp += fp
					statistics[vartype].allele_presence_fn += fn

				# check if genotype predictions are identical
				if callset_gt.get_genotype() == gt.get_genotype():
					statistics[vartype].correct_all += 1
					if gt.nonref:
						statistics[vartype].correct_nonref += 1
				else:
					statistics[vartype].wrong_all += 1
					if gt.nonref:
						statistics[vartype].wrong_nonref += 1
			else:
				statistics[vartype].not_in_callset_all += 1
				if gt.nonref:
					statistics[vartype].not_in_callset_nonref += 1
				if truth_is_biallelic:
					statistics[vartype].confusion_matrix[gt.get_binary_genotype()][4] += 1
					statistics[vartype].not_in_callset_biallelic += 1
					tp, fp, fn = allele_presence_counts(gt, None)
					statistics[vartype].allele_presence_tp += tp
					statistics[vartype].allele_presence_fp += fp
					statistics[vartype].allele_presence_fn += fn

		for vartype in varianttypes:
			assert (
				statistics[vartype].correct_all
				+ statistics[vartype].wrong_all
				+ statistics[vartype].not_typed_all
				== statistics[vartype].total_intersection
			)
		if self._quiet_duplicate_warnings and multiple_prediction_positions:
			sys.stderr.write('Treated ' + str(multiple_prediction_positions) + ' positions with multiple callset predictions as untyped.\n')
		# write statistics and confusion matrices to file
		for i,vartype in enumerate(varianttypes):
			statistics[vartype].print_stats_to_file(tsv_files[i])
			statistics[vartype].print_matrix_to_file(txt_files[i])

if __name__ == "__main__":

	# baseline: the file containing variants and true genotypes
	# callset: contains genotype predictions of genotyper	
	parser = argparse.ArgumentParser(prog='genotype-concordance.py', description=__doc__)
	parser.add_argument('baseline', metavar='BASELINE', help='baseline VCF (ground truth).')
	parser.add_argument('callset', metavar='CALLSET', help='callset VCF (genotyped variants).')
	parser.add_argument('prefix', metavar='OUTFILE', help='prefix of the output file name.')
	parser.add_argument('samples', metavar='SAMPLES', help='comma separated list of samples to evaluate.')
	parser.add_argument('--qualities', default='0', metavar='GQ-THRESHOLDS', help='comma separated list of GQ-thresholds to consider (default: consider all variants regardless of quality).')
	parser.add_argument('--allele-frequencies', default='0', metavar='AF-THRESHOLDS', help='comma separated list of allele frequency thresholds to consider (default: consider all variants regardless of allele frequency.).')
	parser.add_argument('--unique-kmers', default='0', metavar='UK-THRESHOLDS', help='comma separated list of unique kmer counts for a variant (only works for VCFs with UK tag.).')
	parser.add_argument('--missing', default='0', metavar='MISSING-ALLELES_THRESHOLDS', help='comma separated list of missing allele counts for a variant (only works for VCFs with MA tag.).')
	parser.add_argument('--use-qual', default=False, action='store_true', help='use qualities in QUAL field instead of GQ fields (default: use GQ field).')
	parser.add_argument('--bed', default=None, metavar='BED', help='BED file to restrict evaluation to variants within these regions.')
	args = parser.parse_args()

	quality_thresholds = [int(i) for i in args.qualities.split(',')]
	allele_freq_thresholds = [float(i) for i in args.allele_frequencies.split(',')]
	uk_thresholds = [int(i) for i in args.unique_kmers.split(',')]
	missing_thresholds = [int(i) for i in args.missing.split(',')]
	samples = args.samples.split(',')
	bed_regions = parse_bed(args.bed) if args.bed else None
	genotype_concordance = GenotypeConcordanceComputer(args.baseline, args.callset, samples, args.use_qual)


	# compute statistics for each sample
	for sample in samples:
		# prepare output files
		variantnames = []
		for vartype in VariantType:
			name = vartype.name.replace('_', '-')
			variantnames.append(name)
		tsv_files = [open(args.prefix + '_' + sample + '_' + name + '.tsv', 'w') for name in variantnames]
		txt_files = [open(args.prefix + '_' + sample + '_' + name + '.txt', 'w') for name in variantnames]
		# write headers to tsv files
		header = '\t'.join([
				'quality',
				'allele_frequency',
				'unique_kmers',
				'missing_alleles',
				'total_baseline',
				'total_baseline_biallelic',
				'total_baseline_nonref',
				'total_intersection',
				'correct_all',
				'wrong_all',
				'not_typed_all',
				'correct_biallelic',
				'wrong_biallelic',
				'not_typed_biallelic',
				'correct_non-ref',
				'wrong_non-ref',
				'not_typed_non-ref',
				'nr_correct_all',
				'nr_wrong_all',
				'nr_not_typed_all',
				'nr_not_in_callset_all',
				'nr_correct_biallelic',
				'nr_wrong_biallelic',
				'nr_not_typed_biallelic',
				'nr_not_in_callset_biallelic',
				'nr_correct_non-ref',
				'nr_wrong_non-ref',
				'nr_not_typed_non-ref',
				'nr_not_in_callset_non-ref',
				'allele_presence_tp',
				'allele_presence_fp',
				'allele_presence_fn',
	]) + '\n'
		for tsv_file in tsv_files:
			tsv_file.write(header)
		vartypes = [vartype for vartype in VariantType]
		# compute statistics using all variants (regardless of thresholds)
		genotype_concordance.print_statistics(sample, vartypes, 0, 0.0, 0, 0, txt_files, tsv_files, bed_regions=bed_regions)
		# consider different thresholds on number of unique kmers
		for uk_count in uk_thresholds:
			for quality in quality_thresholds:
				if uk_count == quality == 0:
					continue
				genotype_concordance.print_statistics(sample, vartypes, quality, 0.0, uk_count, 0, txt_files, tsv_files, bed_regions=bed_regions)
		# consider different thresholds on allele frequencies
		for allele_freq in allele_freq_thresholds:
			if allele_freq == 0:
				continue
			genotype_concordance.print_statistics(sample, vartypes, 0, allele_freq, 0, 0, txt_files, tsv_files, bed_regions=bed_regions)
		# consider different thresholds on number of missing alleles
		for missing_count in missing_thresholds:
			if missing_count == 0:
				continue
			genotype_concordance.print_statistics(sample, vartypes, 0, 0.0, 0, missing_count, txt_files, tsv_files, bed_regions=bed_regions)
		# close files
		for i in range(len(tsv_files)):
			tsv_files[i].close()
			txt_files[i].close()
