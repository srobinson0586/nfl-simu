import os
import sys
import numpy as np
from numpy.random import choice
from models import train_models
from agents import OptimalAgent, TwoPointAgent, FourthDownConversionAgent, RandomAgent, ConventionalAgent
from win_prob import next_state


def result(team1, team2, print_results=False):
	if team1.score == team2.score:
		if print_results:
			print("It's a tie! Final score:\n %s-%d, %s-%d" % (team1.name, team1.score, team2.name, team2.score))
		return None
	else:
		winner = team1 if team1.score > team2.score else team2
		if print_results:
			loser = team1 if winner == team2 else team2
			print("%s is the winner! Final score:\n %s-%d, %s-%d" % (winner.name, winner.name, winner.score, loser.name, loser.score))
		return winner

if len(sys.argv) == 1:
	print("too few arguments")
	exit()

filename = sys.argv[1]
print("loading data from %s" % filename)
try:
	input_file = open(filename, 'rb')
	win_probability = np.load(input_file)
	input_file.close()
except Exception as e:
	print("unable to load data from %s: %s" % (filename, e.strerror))
	exit()

agent1 = OptimalAgent(win_probability, "agent1")
agent2 = OptimalAgent(win_probability, "agent2")

#models = [om, yd, fg, sm, pm, tr, to, fd]
models = train_models(1)
state = [75, 1800, 1, 0, 0]
pos_team = agent1
def_team = agent2

om = models[0]
yd = models[1]
fg = models[2]
sm = models[3]
pm = models[4]
tr = models[5]
to = models[6]
fd = models[7]

while state[1] > 0 and abs(state[4]) < 100:
	outcome = choice(om.classes, p=om.predict(state))
	# outcome = 'touchdown'
	runoff = choice(tr.classes, p = tr.predict(state)) * 30 + 30
	
	if outcome == 'defensive_touchdown':
		state[4] *= -1
		current_score_diff = state[4]
		state = def_team.make_TD_decision(state, runoff)
		def_team.score += (state[4] * -1 - current_score_diff)
	
	elif outcome == 'end_of_half':
		state[1] = 0



	elif outcome == 'fourth_down':
		yd_result = choice(yd.classes, p=yd.predict(state))
		yards = yd_result + choice(sm.classes, p=sm.predict(yd_result))
		new_position = state[0] - yards

		GTG = False
		if new_position <= 10:
			togo = new_position
			GTG = True
		elif state[0] > new_position:
			#forward progress was made
			togo = 10 - ((state[0] - new_position) % 10)
		else:
			#forward progress was not made
			togo = new_position - (state[0] - 10)

		state[0] = int(new_position)
		current_score_diff = state[4]
		state, keep_pos = pos_team.make_4th_decision(state, togo, fd, fg, pm, runoff, GTG)
		if not keep_pos:
			pos_team.score += (state[4] * -1 - current_score_diff)
			temp = pos_team
			pos_team = def_team
			def_team = pos_team


		
	elif outcome == 'safety':
		state = next_state(state, 75, runoff, -2)
		temp = pos_team
		pos_team = def_team
		def_team = pos_team



	elif outcome == 'touchdown':
		current_score_diff = state[4]
		state = pos_team.make_TD_decision(state, runoff)
		pos_team.score += (state[4] * -1 - current_score_diff)
		temp = pos_team
		pos_team = def_team
		def_team = pos_team



	elif outcome == 'turnover':
		new_position = choice(to.classes, p=to.predict(state)) * 5
		state = next_state(state, new_position, runoff, 0)
		temp = pos_team
		pos_team = def_team
		def_team = pos_team



	else:
		print("unexpected outcome: %s", outcome)
		exit()

result(agent1, agent2, True)