from numpy.random import random, choice
from win_prob import next_state

class Agent():
	def __init__(self):
		self.score = 0
		self.name = self.__class__.__name__

	def make_TD_decision(self, current_state, runoff):
		pass

	def make_4th_decision(self, current_state, togo, fd_model, fg_model, punt_model, runoff, GTG):
		pass
		

class OptimalAgent(Agent):
	def __init__(self, wp):
		self.win_probabilities = wp
		self.score = 0
		self.name = self.__class__.__name__

	def make_TD_decision(self, current_state, runoff):
		#state if the extra point is made
		XP_state = next_state(current_state, 75, runoff, 7)

		#state if a 2 point conversion succeeds
		TP_state = next_state(current_state, 75, runoff, 8)

		#state if PAT fails
		FAIL_state = next_state(current_state,75, runoff, 6)

		#probability of winning if decision is too kick XP
		pw_XP = 0.95 * (1 - self.win_probabilities[tuple(XP_state)]) + 0.05 * (1 - self.win_probabilities[tuple(FAIL_state)])

		#probability of winning if decision is to go for 2 point conversion
		pw_2P = 0.5 * (1 - self.win_probabilities[tuple(TP_state)]) + 0.5 * (1 - self.win_probabilities[tuple(FAIL_state)])
		if self.win_probabilities[tuple(XP_state)] < 0 or self.win_probabilities[tuple(TP_state)] < 0 or self.win_probabilities[tuple(FAIL_state)] < 0:
			print("FUCK")
		result = random()
		if pw_XP > pw_2P and result <= 0.95:
			return XP_state
		elif pw_2P >= pw_XP and result <= 0.5:
			return TP_state
		else:
			return FAIL_state


	def make_4th_decision(self, current_state, togo, fd_model, fg_model, punt_model, runoff, GTG):
		FG_state = next_state(current_state, 75, runoff, 3)
		GO_state = self.make_TD_decision(current_state, runoff) if GTG else next_state(current_state, current_state[0] - togo, runoff, 0, True)
		FAIL_state = next_state(current_state, 100 - current_state[0], runoff, 0)

		pw_PUNT = 0.0
		punt_predictions = punt_model.predict(current_state[0])
		for p in range(0,len(punt_model.classes)):
			if punt_predictions[p] > 0:
				new_state = next_state(current_state, punt_model.classes[p] * 5 + 5, runoff, 0)
				pw_PUNT += punt_predictions[p] * (1 - self.win_probabilities[tuple(new_state)])

		pw_FG = fg_model.predict(current_state[0] + 17) * (1 - self.win_probabilities[tuple(FG_state)]) + \
				(1 - fg_model.predict(current_state[0] + 17)) * (1 - self.win_probabilities[tuple(FAIL_state)])

		if self.win_probabilities[tuple(FAIL_state)] < 0 or self.win_probabilities[tuple(FG_state)] < 0 or self.win_probabilities[tuple(GO_state)] < 0:
			print("FUCK")
		#if goal to go, then converting on 4th will result in a touchdown
		if GTG:
	  		#state if the extra point is made
			XP_state = next_state(current_state,75, runoff, 7)

			#state if a 2 point conversion succeeds
			TP_state = next_state(current_state, 75, runoff, 8)

			#state if PAT fails
			PAT_FAIL_state = next_state(current_state,75, runoff, 6)
			if self.win_probabilities[tuple(XP_state)] < 0 or self.win_probabilities[tuple(TP_state)] < 0 or self.win_probabilities[tuple(PAT_FAIL_state)] < 0:
				print("FUCK")

			#probability of winning if decision is too kick XP
			pw_XP = 0.95 * (1 - self.win_probabilities[tuple(XP_state)]) + 0.05 * (1 - self.win_probabilities[tuple(PAT_FAIL_state)])

			#probability of winning if decision is to go for 2 point conversion
			pw_2P = 1 - (0.5 * self.win_probabilities[tuple(TP_state)]) + 0.5 * (1 - self.win_probabilities[tuple(PAT_FAIL_state)])

			#probabilty of winning if decision is to attempt a 4th down conversion
			pw_GO = fd_model.predict(togo)[1] * max(pw_XP, pw_2P) + fd_model.predict(togo)[0] * (1 - self.win_probabilities[tuple(FAIL_state)])

			#going for it maxmizes win probability
			if max(pw_GO, pw_PUNT, pw_FG) == pw_GO:
				conversion_result = random()
				if conversion_result <= fd_model.predict(togo)[1]:
					PAT_result = random()
					if pw_XP > pw_2P and PAT_result <= 0.95:
						return XP_state, False
					elif pw_2P >= pw_XP and PAT_result <= 0.5:
						return TP_state, False
					else:
						return PAT_FAIL_state, False

		else:

			#probability of winning if decision is to attempt a 4th down conversion
			pw_GO = fd_model.predict(togo)[1] * self.win_probabilities[tuple(GO_state)] + fd_model.predict(togo)[0] * (1 - self.win_probabilities[tuple(FAIL_state)])

			if max(pw_GO, pw_PUNT, pw_FG) == pw_GO:
				conversion_result = random()
				if conversion_result <= fd_model.predict(togo)[1]:
					return GO_state, True
				else:
					return FAIL_state, False

		
		#kicking a FG maximizes win probability
		if max(pw_GO, pw_PUNT, pw_FG) == pw_FG:
			if random() <= fg_model.predict(current_state[0] + 17):
				return FG_state, False
			else:
				return FAIL_state, False

		#punting maximizes win probability 
		else:
			punt_result = choice(punt_model.classes, p=punt_predictions)
			PUNT_state = next_state(current_state, punt_result * 5 + 5, runoff, 0)
			return PUNT_state, False

class RiskyAgent(Agent):
	def make_TD_decision(self, current_state, runoff):

		#state if a 2 point conversion succeeds
		TP_state = next_state(current_state, 75, runoff, 8)

		#state if PAT fails
		FAIL_state = next_state(current_state,75, runoff, 6)

		result = random()
		if result <= 0.5:
			return TP_state
		else:
			return FAIL_state

	def make_4th_decision(self, current_state, togo, fd_model, fg_model, punt_model, runoff, GTG):
		#if goal to go, then converting on 4th will result in a touchdown
		FAIL_state = next_state(current_state, 100 - current_state[0], runoff, 0)
		GO_state = self.make_TD_decision(current_state, runoff) if GTG else next_state(current_state, current_state[0] - togo, runoff, 0, True)
		
		conversion_result = random()
		if conversion_result <= fd_model.predict(togo)[1]:
			return GO_state, not GTG
		else:
			return FAIL_state, False


class RandomAgent(Agent):
	def make_TD_decision(self, current_state, runoff):
		#state if the extra point is made
		XP_state = next_state(current_state, 75, runoff, 7)

		#state if a 2 point conversion succeeds
		TP_state = next_state(current_state, 75, runoff, 8)

		#state if PAT fails
		FAIL_state = next_state(current_state,75, runoff, 6)


		decision = choice([1,2])
		result = random()
		if decision == 1 and result <= 0.95:
			return XP_state
		elif decision == 2 and result <= 0.5:
			return TP_state
		else:
			return FAIL_state

	def make_4th_decision(self, current_state, togo, fd_model, fg_model, punt_model, runoff, GTG):
		FG_state = next_state(current_state, 75, runoff, 3)
		GO_state = self.make_TD_decision(current_state, runoff) if GTG else next_state(current_state, current_state[0] - togo, runoff, 0, True)
		FAIL_state = next_state(current_state, 100 - current_state[0], runoff, 0)
		punt_result = choice(punt_model.classes, p=punt_model.predict(current_state[0]))
		PUNT_state = next_state(current_state, punt_result * 5 + 5, runoff, 0)
		
		decision = choice(['FG', 'punt', 'go'])
		result = random()
		if decision == 'FG' and result <= fg_model.predict(current_state[0] + 17):
			return FG_state, False
		elif decision == 'punt':
			return PUNT_state, False
		elif result <= fd_model.predict(togo)[1]:
			return GO_state, not GTG
		else:
			return FAIL_state, False


class ConventionalAgent(Agent):
	def make_TD_decision(self, current_state, runoff):
		#state if the extra point is made
		XP_state = next_state(current_state, 75, runoff, 7)

		#state if a 2 point conversion succeeds
		TP_state = next_state(current_state, 75, runoff, 8)

		#state if PAT fails
		FAIL_state = next_state(current_state,75, runoff, 6)

		
		new_score_diff = current_state[4] + 6
		
		go_for_two = new_score_diff in {-15, -13, -11, -10, -8, -5, -4, -2, 1, 5, 12}


		result = random()
		if not go_for_two and result <= 0.95:
			return XP_state
		elif go_for_two and result <= 0.5:
			return TP_state
		else:
			return FAIL_state


	def make_4th_decision(self, current_state, togo, fd_model, fg_model, punt_model, runoff, GTG):
		FG_state = next_state(current_state, 75, runoff, 3)
		GO_state = self.make_TD_decision(current_state, runoff) if GTG else next_state(current_state, current_state[0] - togo, runoff, 0, True)
		FAIL_state = next_state(current_state, 100 - current_state[0], runoff, 0)
		punt_result = choice(punt_model.classes, p=punt_model.predict(current_state[0]))
		PUNT_state = next_state(current_state, punt_result * 5 + 5, runoff, 0)

		field_position = current_state[0]
		in_FG_range = field_position <= 37
	
		if togo == 1 and field_position <= 91:
			attempt_conversion = True
		elif togo == 2 and field_position <= 72:
			attempt_conversion = True
		elif togo == 3 and field_position <= 60:
			attempt_conversion = True
		elif togo == 4 and field_position <= 55 and field_position >= 29:
			attempt_conversion = True 
		elif togo == 5 and field_position <= 50 and field_position >= 33:
			attempt_conversion = True
		elif togo == 6 and field_position <= 47 and field_position >= 35:
			attempt_conversion = True
		elif togo == 7 and field_position <= 44 and field_position >= 36:
			attempt_conversion = True
		elif togo == 8 and field_position <= 41 and field_position >= 37:
			attempt_conversion = True
		elif togo == 9 and field_position == 38:
			attempt_conversion = True
		else:
			attempt_conversion = False

		punt = not attempt_conversion and not in_FG_range
		result = random()
		if attempt_conversion and result <= fd_model.predict(togo)[1]:
			return GO_state, not GTG
		elif in_FG_range and result <= fg_model.predict(field_position + 17):
			return FG_state, False
		elif punt:
			return PUNT_state, False
		else:
			return FAIL_state, True













