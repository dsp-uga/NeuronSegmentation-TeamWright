
from glob import glob

def merge_json(args):

	output = args.output
	if output[len(output) - 1] != '/':
		output += '/'

	files = sorted(glob(output + '*.json'))

	f = open("submit.json", 'w')

	f.write('[')
	n = len(files)
	for i in range(n):
		o = open(files[i], 'r')
		s = o.readline()
		f.write(s)
		if i != n-1:
			f.write(',\n')
		o.close()
	f.write(']')
	f.close()
