
from src.nmf import nmf
from merge_json import merge_json as mj
import argparse

parser = argparse.ArgumentParser(description='CSCI 8360 - Project 3, Team Wright')

parser.add_argument("-d", "--dataset", default = "dataset",
					help = "Path to dataset containing a neurofinder.**.**.test folders\n"
							"Subfolders should be in the format mentioned, as the ids are parsed from these names")

parser.add_argument("-o", "--output", default = "output",
					help = "Path to directory where output will be written")


args = parser.parse_args()
nmf(args)
mj(args)
