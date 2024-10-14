# QT-WEAVER
#### QT-WEAVER (version 1.0)
This repository contains the official implementation of [***QT-WEAVER: Correcting quartet distribution improves phylogenomic analyses despite gene tree estimation error***]

## Short Description

**QT-WEAVER** is a quartet distribution correcition heuristic for improving species trees estimation from quartets. 

QT-WEAVER is a heuristic method that corrects an weighted quartet distribution based on conflict-scores.

+ Input: A weighted quartet distribution file
+ Output: A corrected version of the weighted quartet distribution file

## Installing QT-WEAVER

Before installing QT-WEAVER, please sure that you have the following programs installed:

- [Python](https://www.python.org/downloads/): Version >= 3.7
- [Pip](https://pip.pypa.io/en/stable/installation/): Version >= 21.0
<!-- - [Java](https://www.oracle.com/java/technologies/downloads/): Version >= 11.0 (if you want to generate the species trees using [wQFM](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btab428/6292084)) -->

To install the required python packages, use the following command

```bash
pip install -r requirements.txt
``` 

The authors recommend installing [Anacoda](https://www.anaconda.com/) and using a seperate [conda environment](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) to install QT-WEAVER.

<!-- If you use wQFM, please cite the paper ["wQFM: Highly Accurate Genome-scale Species Tree Estimation from Weighted Quartets"](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btab428/6292084). -->

## Usage

####  For correcting the weighted quartets distribution, use -i and -o flag.
   
<!-- Code Blocks -->
```bash
python QT-WEAVER.py -i <input-quartet-distribution-file> -o <output-quartet-distribution-file>
``` 
#### To set the weighting scheme and the conflict set size, use the -w and -m flags respectively. 
```bash
python QT-WEAVER.py -i <input-quartet-distribution-file> -o <output-quartet-distribution-file> -w <weighting> -m <mode>
``` 

## Example

Two quartet distribution files have been provided in the repository.

```bash
python QT-WEAVER.py -i data/test_15_taxa.wqrts -o data/test_15_taxa_corrected.wqrts
``` 