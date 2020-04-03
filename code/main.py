import os
from models import OutcomeModel, YardDistributionModel, TimeRunoffModel, FieldGoalModel, SmallYardDistributionModel, PuntModel, TurnoverFieldPosModel

os.environ['CUDA_VISIBLE_DEVICES'] = ""
# om = OutcomeModel()
# om.train_model(1)
# print(om.predict([25,1800,1,1,0]))

# yd = YardDistributionModel()
# yd.train_model(1000)

# # om.generate_buckets()
# # om.evaluate()
# # yd.generate_buckets()
# # yd.evaluate()

# fg = FieldGoalModel()
# fg.train_model()
# # fg.evaluate()

# sm = SmallYardDistributionModel()
# sm.train_model()
# # sm.evaluate()

# pm = PuntModel()
# pm.train_model(50)
# #pm.evaluate()

tr = TimeRunoffModel()
print(tr.classes)
# tr.train_model(1000)
# # tr.generate_buckets()
# # tr.evaluate()

to = TurnoverFieldPosModel()
print(to.classes)
# to.train_model(1000)
# # to.generate_buckets()
# # to.evaluate()

# om.evaluate()
# yd.evaluate()
# fg.evaluate()
# sm.evaluate()
# pm.evaluate()
# tr.evaluate()
# to.evaluate()


#current_state =  'field_pos', 'time_remaining', 'is_half_one', 'is_half_two', 'score_differential'

def next_state(current_state, new_field_pos, runoff, score, keep_pos=False):
	#get rid of invalid states
	if new_field_pos <= 0 or new_field_pos >= 100:
		return None
	new_time = current_state[1] - runoff
	if new_time <= -30:
		return None
	new_time = max(0, new_time)
	is_half_one = 1 if current_state[2] == 1 and new_time > 0 else 0
	is_half_two = 1 if is_half_one == 0 else 0
	if new_time == 0:
		new_time = 1800
	new_score = current_state[4] + score
	temp = pos_team
	pos_team = int(not pos_team)
	if new_time == 1800 and is_half_two == 1:
		pos_team = 1
	elif keep_pos:
		pos_team = temp
	if temp != pos_team:
		new_score *= -1

	return [new_field_pos, new_time, is_half_one, is_half_two, new_score]

example = [25, 1800, 1, 0, 0]
example1 = [50, 10, 1, 0, 7]
example2 = [30, 100, 0, 1, -7]


# #holds all the possible states
win_probability = np.full((100,1801,2,2,120), -1.0)

pos_team = 0
inital_state = [25, 1800, 1, 0, 0]

# def prob(current_state):
# 	if not current_state:
# 		#current state must be invalid
# 		return 0.0
#  	if win_probability[tuple(current_state)] > -1.0:
#  		return win_probability[tuple(current_state)]
#  	#time_remaining = 0
#  	else if current_state[1] == 0:
#  		if current_state[4] > 0:			
#  			win_probability[tuple(current_state)] = 1.0
#  		elif current_state[4] < 0:
#  			win_probability[tuple(current_state)] = 0.0
#  		else:
#  			win_probability[tuple(current_state)] = 0.5
 	
#  	else:
#  		#oh boy
#  		total = 0.0
#  		runoff_predictions = tr.predict(current_state)
#  		for i in range(0, len(tr.classes)):
#  			runoff = int(tr.classes[i])

#  			pw_2P = 0.5 * prob(next_state(current_state, 75, runoff, 8)) + 0.5 * prob(next_state(current_state, 75, runoff, 6))
#  			pw_XP = 0.95 * prob(next_state(current_state,75, runoff, 7)) + 0.05 * prob(next_state(current_state, 75, runoff, 6))
#  			pw_TD = max(pw_2P, pw_XP)
 			
#  			turnover_predictions = to.predict(current_state)
#  			turnover_predictions = [prob(current_state, turnover_predictions[i])]
#  			pw_TO = 




#  			# if current_state[0] > 10:
#  			# 	pw_GO = 0.5 * prob(next_state(current_state, current_state[0] - 10, runoff, 0, True)) + 0.5 * prob(next_state(current_state, 100 - current_state[0], runoff, 0))
#  			# else:
#  			# 	pw_GO = 0.5 * max(pw_XP, pw_2P) + 0.5 * prob(next_state(current_state, 100 - current_state[0], runoff, 0))

#  			# pw_FG = 



 		















