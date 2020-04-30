import os
import sys
import numpy as np
from numpy.random import choice
from models import train_models
from agents import OptimalAgent, RiskyAgent, RandomAgent, ConventionalAgent
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

def simulate(agent1, agent2, models, trials):

	agents = (agent1, agent2)


	#models = [om, yd, fg, sm, pm, tr, to, fd]
	om = models[0]
	yd = models[1]
	fg = models[2]
	sm = models[3]
	pm = models[4]
	tr = models[5]
	to = models[6]
	fd = models[7]

	wins = 0
	ties = 0
	losses = 0
	for i in range(trials):
		agents[0].score = 0
		agents[1].score = 0
		for j in range(2):
			state = [75, 1800, 1, 0, 0]
			possession = j
			offense = agents[possession]
			while state[1] > 0 and abs(state[4]) < 100:
				outcome = choice(om.classes, p=om.predict(state))
				runoff = choice(tr.classes, p=tr.predict(state)) * 30 + 30
				if outcome == 'defensive_touchdown':
					state[4] *= -1
					current_score_diff = state[4]
					defense = agents[not possession]
					state = defense.make_TD_decision(state, runoff)
					defense.score += (state[4] * -1 - current_score_diff)
				
				elif outcome == 'end_of_half':
					state[1] = 0

				elif outcome == 'fourth_down':
					new_position = 0
					while new_position <= 0 or new_position >= 100:
						yd_result = choice(yd.classes, p=yd.predict(state))
						yards = yd_result * 10 + choice(sm.classes, p=sm.predict(yd_result))
						new_position = state[0] - yards

					GTG = False
					if state[0] <= 10 or new_position <= 10:
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
					state, keep_pos = offense.make_4th_decision(state, togo, fd, fg, pm, runoff, GTG)
					if not keep_pos:
						offense.score += (state[4] * -1 - current_score_diff)
						possession = not possession
						offense = agents[possession]


				
				elif outcome == 'safety':
					state = next_state(state, 75, runoff, -2)
					possession = not possession
					offense = agents[possession]
					offense.score += 2



				elif outcome == 'touchdown':
					current_score_diff = state[4]
					state = offense.make_TD_decision(state, runoff)
					offense.score += (state[4] * -1 - current_score_diff)
					possession = not possession
					offense = agents[possession]



				elif outcome == 'turnover':
					new_position = max(choice(to.classes, p=to.predict(state)) * 5, 5)
					state = next_state(state, new_position, runoff, 0)
					possession = not possession
					offense = agents[possession]

				else:
					print("unexpected outcome: %s", outcome)
					exit()

				if state == None:
					print(new_position)
					print(outcome)

		winner = result(agent1, agent2)
		if winner == agent1:
			wins += 1
		elif winner == agent2:
			losses += 1
		else:
			ties += 1

	return wins, ties, losses

if __name__ == '__main__':

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

	models = train_models(1000)
	opponents = [OptimalAgent(win_probability), RiskyAgent(), RandomAgent(), ConventionalAgent()]
	
	my_agent = OptimalAgent(win_probability)
	for o in opponents:	
		wins, ties, losses = simulate(my_agent, o, models, 1000)
		print("%s vs. %s - wins: %d, ties: %d, losses:%d" % (my_agent.name, o.name, wins, ties, losses))

	# my_agent = RandomAgent()
	# for o in opponents:	
	# 	wins, ties, losses = simulate(my_agent, o, models, 1000)
	# 	print("%s vs. %s - wins: %d, ties: %d, losses:%d" % (my_agent.name, o.name, wins, ties, losses))

	# my_agent = ConventionalAgent()
	# for o in opponents:	
	# 	wins, ties, losses = simulate(my_agent, o, models, 1000)
	# 	print("%s vs. %s - wins: %d, ties: %d, losses:%d" % (my_agent.name, o.name, wins, ties, losses))

	# my_agent = RiskyAgent()
	# for o in opponents:	
	# 	wins, ties, losses = simulate(my_agent, o, models, 1000)
	# 	print("%s vs. %s - wins: %d, ties: %d, losses:%d" % (my_agent.name, o.name, wins, ties, losses))











