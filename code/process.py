import pandas as pd
import numpy as np
import csv
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def generate_data(no_score_only=False):
	data = pd.read_csv("../data/drive_data.csv")
	data = data.drop(columns=['game_id', 'drive_number'])
	if no_score_only:
		data = data[data['result'].isin(['fourth_down'])]
	encoder = LabelEncoder()
	y = data['yds_gained'] if no_score_only else data['result']
	encoder.fit(y)
	encoded_y = encoder.transform(y)
	labels = np_utils.to_categorical(encoded_y)
	data.insert(data.shape[1], "y", list(labels), False)
	data = data.drop(columns=['result', 'yds_gained'])
	return data

if __name__ == '__main__':	
	touchdown = 145
	field_goal_result = 39
	half_seconds = 11
	interception = 120
	fumble_lost = 138
	safety = 135
	field_position = 8
	game_half = 13
	score_differential = 54 
	pos_tol = 48
	def_tol = 49
	play_type = 25
	fourth_down_failed = 117

	with open('../data/drive_data.csv', mode='w') as data_file:
		writer = csv.writer(data_file, delimiter=',', quotechar='"')
		writer.writerow(['game_id', 'drive_number', 'field_pos', 'time_remaining', 'is_half_one', 'is_half_two', 'UTM', 'score_differential', 'pos_team TOL', 'def_time TOL', 'result', 'yds_gained'])
		for filename in os.listdir('../data'):

			if filename != "drive_data.csv" and filename[0] != '.':
				print(filename)
				with open("../data/" + filename) as current_data:
					reader = csv.reader(current_data, delimiter=',', quotechar='"')
					line = 1
					current_drive = 0
					result = ""
					begin = False
					current_game = ''
					for row in reader:
						
						if line == 1 or (row[25] == 'NA' and row[half_seconds] != '0') :
							line += 1
							continue

						if row[1] != current_game:
							current_game = row[1]
							current_drive = 0
							begin = False

						elif row[half_seconds] == '0':
							if begin:
								begin = False
							else:
								writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'end_of_half', 'NA'])
							if is_half_two == '1':
								current_drive = 0
							line += 1
							continue

						if begin:
							current_game = row[1]
							current_drive += 1
							game_id = row[1] + ' - ' + str(line)
							#print(game_id)
							field_pos = row[field_position]
							time_remaining = row[half_seconds]
							is_UTM = '1' if int(time_remaining) <= 120 else '0'
							is_half_one = '1' if row[game_half] == 'Half1' else '0'
							is_half_two = '1' if row[game_half] == 'Half2' else '0'
							score_diff = row[score_differential]
							p_timeout = row[pos_tol]
							d_timeout = row[def_tol]
							begin = False

						if current_drive == 0:
							begin = True
							line += 1
							continue

						if row[play_type] == 'kickoff' and not row[touchdown] == '1':
							begin = True
							
						elif row[play_type] == 'punt' or row[fourth_down_failed] == '1' or row[field_goal_result] == 'missed':
							if not row[touchdown] == '1':
								begin = True
							writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'fourth_down', int((int(field_pos) - int(row[field_position]))/10)])

						elif row[touchdown] == '1':
							if row[interception] == '1' or row[fumble_lost] == '1':
								result = 'defensive_touchdown'
							else:
								result = 'touchdown'
							writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, result, field_pos])

						elif row[safety] == '1':
							writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'safety', 'NA'])

						elif row[interception] == '1' or row[fumble_lost] == '1':
							writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'turnover', 'NA'])
							begin = True

						elif row[field_goal_result] == 'made':
							writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'fourth_down', int((int(field_pos) - int(row[field_position]))/10)]) 

		
							
						line += 1





