"""
would need to install following packages
- numpy
- scipy
- scikit-learn
- thunder-python
- thunder-extraction
"""

from glob import glob
import thunder as td
from extraction import NMF
import json
import os

def nmf(args):

	dataset = args.dataset
	if dataset[len(dataset)-1] != '/':
		dataset += '/'
	dirs = sorted(glob(dataset + '*/'))

	for d in dirs:

		print("Working on folder", d)

		# Read images from the dataset
		path = d + 'images/'
		V = td.images.fromtif(path, ext = 'tiff')

		# Applying NMF on data
		algorithm = NMF(k = 10, max_iter = 30, percentile = 99)
		model = algorithm.fit(V, chunk_size = (50,50), padding = (25, 25))
		merged = model.merge(overlap = 0.1, max_iter = 2, k_nearest = 5)

		# extracting ROI
		roi = [{'coordinates': r.coordinates.tolist()} for r in merged.regions]
		
		# converting to json format
		dataset = d[d.find('neurofinder.') + 12 : d.find('neurofinder.') + d[d.find('neurofinder.'):].find('/')]
		json_string = {'dataset': dataset, 'regions': roi}

		# writing to json file
		output = args.output
		if not os.path.exists(output):
			os.makedirs(output)
		if output[len(output)-1] != '/':
			output += '/'
		f = open(output + dataset + '.json', 'w')
		f.write(json.dumps(json_string))
		f.close()
