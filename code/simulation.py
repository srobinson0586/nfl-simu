import os
from numpy.random import choice
from models import train_models
from agents import OptimalAgent


filename = sys.argv[1]
if !os.path.isfile(filename):
	print("%s does not exist" % filename)
else:
	input_file = open(filename, 'rb')
	try:
		win_probability = np.load(input_file)
	except:
		print("unable to load data from %s" % filename)

agent1 = OptimalAgent(win_probability)
agent2 = OptimalAgent(win_probability)

models = train_models(1)
state = [75, 1800, 1, 0, 0]