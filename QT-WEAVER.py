from lib.Quartet import Quartet
import random
from copy import deepcopy
import sys
import numpy as np
from tqdm import tqdm
import argparse
import time

MAX_WEIGHT = 1e8
eps = 1e-10 

input_conflict_score = 0

def parse_arguments():
    parser = argparse.ArgumentParser(prog='QT-WEAVER.py', description='QT-WEAVER: Quartet distribution correction based on conflict score')
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
    	'-i', '--input',
    	help="Input quartet distribution file path",
    	dest="input_file",
        type=str,
    	metavar="INPUT_FILE",
        required=True
        )

    required.add_argument("-o", "--output",
        help="Output quartet distribution file path",
        dest="output_file",
        type=str,
        metavar="OUTPUT_FILE",
        required=True
        )
    
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('-w', '--weighting',
		help="Weighting scheme for conflict scores [default: %(default)s] [choices: %(choices)s]",
		choices={"min", "regular"},
		dest='weighting',
        type=str,
        metavar="WEIGHTING_SCHEME",
        default="min"
        )

    optional.add_argument('-m', '--mode',
		help="How many conflicting sets to consider for each quartet topology [default: %(default)s] [choices: %(choices)s]",
		choices={"six", "all"},
		dest="mode",
        type=str,
        metavar="MODE",
        default="all"
        )
		  
    args = parser.parse_args()

    return args

def get_conflict_set_dict(mode="all"):
    dct = {}

    if mode == "all":
        dct[(0,2,1,4)] = [(0,1,3,4), (0,4,1,3), (0,2,3,4), (0,3,2,4), (1,2,3,4), (1,3,2,4)]
        dct[(0,3,1,4)] = [(0,1,2,4), (0,4,1,2), (0,2,3,4), (0,3,2,4), (1,2,3,4), (1,3,2,4)]
        dct[(0,4,1,2)] = [(0,1,3,4), (0,2,3,4), (0,3,2,4), (1,2,3,4), (1,3,2,4)]
        dct[(0,4,1,3)] = [(0,1,2,4), (0,2,3,4), (0,3,2,4), (1,2,3,4), (1,3,2,4)]
        dct[(0,3,2,4)] = [(1,2,3,4), (1,4,2,3)]
        dct[(1,3,2,4)] = [(0,2,3,4), (0,4,2,3)]
        dct[(0,2,3,4)] = [(1,4,2,3)]
        dct[(1,2,3,4)] = [(0,4,2,3)]
    elif mode == "six":
        dct[(0,2,1,4)] = [(0,4,1,3), (0,1,3,4)]
        dct[(0,4,1,2)] = [(0,3,1,4)]
        dct[(0,2,3,4)] = [(1,3,2,4), (1,4,2,3)]
        dct[(0,3,2,4)] = [(1,2,3,4)]

    return dct

def adjust(input_file, output_file, weighting="min", mode="all", balance_factor=0):
    start_time = time.time()
    print(f"Correcting {input_file} (weighting: {weighting}, mode: {mode})")

    global input_conflict_score
    input_conflict_score = 0

    taxa = set()
    covered = {} # quartets that have been covered

    eq = {} # input quartets
    with open(input_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            l = line.split()
            temp = Quartet(l[0])
            eq[temp] = float(l[1])
            taxa.update(temp.taxa)
            covered[temp] = False

    cq = deepcopy(eq)
    dom_q = []
    weights = [] # list of all quartets with their weights
    wq = {} # used to hold newly computed weights
    changes = 0

    conflict_sets = get_conflict_set_dict(mode)

    for q in tqdm(cq):
        if covered[q]:
            continue

        q3 = [q]
        q3.extend(q.generate_variants())

        total_score = 0
        total_quartets = sum([eq[qq] if qq in eq else 0 for qq in q3])
        for qq in q3:
            if qq not in eq:
                eq[qq] = 0.0
            covered[qq] = True

            # score = find_score(qq, taxa, eq)
            wq[qq] = find_score_conflict(qq, taxa, eq, weighting=weighting, conflict_sets=conflict_sets, balance_value=total_quartets*balance_factor)
            total_score += wq[qq]

        for qq in q3:
            if weighting == "denorm":
                weights.append((qq.des, wq[qq]))
            else:
                weights.append((qq.des, wq[qq]/total_score*total_quartets))

    # write in a file
    with open(output_file, "w") as fp:
        for q in weights:
            fp.write(q[0] + " " + str(q[1]) + "\n")

    conflict_set_size = 28 if "all" in mode else 6
    
    print(f"Finished in {time.time()-start_time} seconds.")

    return input_conflict_score / conflict_set_size

def harmonic_mean(w1, w2):
    if w1 == 0 or w2 == 0:
        return 0
    
    return 2 / (1/w1 + 1/w2)

def find_score_conflict(quartet, taxa, eq, weighting, conflict_sets, balance_value=0):
    global input_conflict_score
    score = 0

    for taxon in taxa:
        if taxon in quartet.taxa:
            continue

        qtaxa = quartet.taxa.copy()
        qtaxa.append(taxon)

        dct = {}
        conflict_set_size = 0

        for k1 in conflict_sets:
            q1 = Quartet.from_taxa(qtaxa[k1[0]], qtaxa[k1[1]], qtaxa[k1[2]], qtaxa[k1[3]])
            dct[q1] = []
            for k2 in conflict_sets[k1]:
                q2 = Quartet.from_taxa(qtaxa[k2[0]], qtaxa[k2[1]], qtaxa[k2[2]], qtaxa[k2[3]])
                dct[q1].append(q2)
                conflict_set_size += 1

        for q1 in dct:
            for q2 in dct[q1]:
                sq1 = eq[q1] if q1 in eq else 0
                sq2 = eq[q2] if q2 in eq else 0
                if weighting == "regular":
                    score += sq1 * sq2
                elif weighting == "min":
                    score += min(sq1, sq2)
                elif weighting == "min2":
                    score += min(sq1, sq2) ** 2

                input_conflict_score += min(eq[quartet], sq1, sq2)
   
    if weighting == "reciprocal":
        return eq[quartet] * score
    else:
        score /= conflict_set_size * (len(taxa)-4)

        adjusted_weight = (eq[quartet] + balance_value) / score if score > eps else MAX_WEIGHT
        adjusted_weight = MAX_WEIGHT if np.isnan(adjusted_weight) else adjusted_weight

        return adjusted_weight


args = parse_arguments()

adjust(args.input_file, args.output_file, weighting=args.weighting, mode=args.mode, balance_factor=0)
