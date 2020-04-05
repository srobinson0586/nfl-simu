import pandas as pd
import numpy as np
import csv
import os


def generate_data(variant):
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    drop = ['result', 'yds_gained / 10', 'actual_yds_gained', 'field_goal_distance', 'field_goal_attempt',
                         'field_goal_distance', 'punt_attempt', 'punt_distance_from_goalline', 'field_pos_at_punt', 'turnover_delta_field_pos', 'time_runoff']
    data = pd.read_csv("../data/drive_data.csv")
    data = data.drop(columns=['game_id', 'drive_number', 'UTM', 'pos_team TOL', 'def_team TOL'])
    classes = None
    if variant == "YardDistributionModel":
        data = data[data['result'].isin(['fourth_down'])]
        y = data['yds_gained / 10']
    elif variant == "SmallYardDistributionModel":
        data = data[data['result'].isin(['fourth_down'])]
        y = data['actual_yds_gained']
    elif variant == 'FieldGoalModel':
        data = data[data['field_goal_attempt'].isin(['1'])]
        y = data['field_goal_distance']
    elif variant == 'PuntModel':
        data = data[data['punt_attempt'].isin(['1'])]
        data = data[data['punt_distance_from_goalline'] <= 18]
    
        y = data['punt_distance_from_goalline']
        data = data.drop(columns=['field_pos', 'time_remaining', 'is_half_one', 'is_half_two', 'score_differential'])
        drop = ['result', 'yds_gained / 10', 'actual_yds_gained', 'field_goal_distance', 'field_goal_attempt',
                         'field_goal_distance', 'punt_attempt', 'punt_distance_from_goalline', 'turnover_delta_field_pos', 'time_runoff']
    elif variant == 'TimeRunoffModel':
    
        data = data[(data['time_runoff'] <= 20) & (data['time_runoff'] >= 0)]
#         dummies = pd.get_dummies(data['result'])
#         data = pd.concat([data, dummies], axis=1)
        y = data['time_runoff']
        
    elif variant == 'TurnoverFieldPosModel':
        data = data[data['result'].isin(['turnover'])]
        y = data['turnover_delta_field_pos']
       
    else:
        y = data['result']
        
    if variant != 'PuntModel' and variant != 'SmallYardDistributionModel' and variant != 'FieldGoalModel':
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)           
        labels = np_utils.to_categorical(encoded_y)
        y = list(labels)
        classes = encoder.classes_

    data.insert(data.shape[1], "y", y, False)
    data = data.drop(columns=drop)
    return data, classes


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
        writer.writerow(['game_id', 'drive_number', 'field_pos', 'time_remaining', 'is_half_one', 'is_half_two', 'UTM', 'score_differential', 'pos_team TOL', 'def_team TOL', 'result', 'yds_gained / 10', 'actual_yds_gained', 'field_goal_attempt', 'field_goal_distance', 'punt_attempt', 'field_pos_at_punt', 'punt_distance_from_goalline', 'turnover_delta_field_pos', 'time_runoff'])
        for filename in os.listdir('../data'):

            if filename != "drive_data.csv" and filename[0] != '.':
                print(filename)
                with open("../data/" + filename) as current_data:
                    reader = csv.reader(
                        current_data, delimiter=',', quotechar='"')
                    line = 1
                    current_drive = 0
                    result = ""
                    begin = True
                    wait_on_field_pos = False
                    current_game = ''
                    for row in reader:

                        if wait_on_field_pos:
                          
                            if row[field_position] == 'NA':
                                line += 1
                                continue
                            if current_row[10] == 'fourth_down':
                               
                                current_row[17] = int((100 - int(row[field_position])) / 5)
                                  
                            else:
                                current_row[18] = int(int(row[field_position]) / 5)
                            writer.writerow(current_row)
                        
                            wait_on_field_pos = False

                        if line == 1 or (row[25] == 'NA' and row[half_seconds] != '0'):
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
                                writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff,
                                                p_timeout, d_timeout, 'end_of_half', 'NA', 'NA', '0', 'NA', '0', 'NA', 'NA', 'NA', int(time_remaining / 30)])
                            if is_half_two == '1':
                                current_drive = 0
                            line += 1
                            continue

                        if begin:
                            current_game = row[1]
                            current_drive += 1
                            game_id = row[1] + ' - ' + str(line)
                            field_pos = int(row[field_position])
                            time_remaining = int(row[half_seconds])
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

                        elif row[fourth_down_failed] == '1':
                            if not row[touchdown] == '1':
                                begin = True
                            writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'fourth_down', int(
                                (int(field_pos) - int(row[field_position]))/10), int(field_pos) - int(row[field_position]), '0', 'NA', '0', 'NA', 'NA', 'NA', int((time_remaining - int(row[half_seconds]))/30)])
                        elif row[play_type] == 'punt':
                            if not row[touchdown] == '1':
                                begin = True
                                wait_on_field_pos = True
                                current_row = [game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'fourth_down', int((int(field_pos) - int(row[field_position]))/10), int(field_pos) - int(row[field_position]) ,'0', 'NA', '1', row[field_position], 0, 'NA', int((time_remaining - int(row[half_seconds]))/30)]
                                


                        elif row[touchdown] == '1':
                            if row[interception] == '1' or row[fumble_lost] == '1':
                                result = 'defensive_touchdown'
                            else:
                                result = 'touchdown'
                            writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout,
                                            d_timeout, result, 'NA', 'NA', '0', 'NA', '0', 'NA', 'NA', 'NA', int(int((time_remaining - int(row[half_seconds]))/30))])

                        elif row[safety] == '1':
                            writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'safety', 'NA', 'NA', '0', 'NA', '0', 'NA', 'NA', 'NA', int(int((time_remaining - int(row[half_seconds]))/30))])

                        elif row[interception] == '1' or row[fumble_lost] == '1':
                            current_row = [game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'turnover', 'NA', 'NA', '0', 'NA', '0', 'NA', 'NA', 'NA', int((time_remaining - int(row[half_seconds]))/30)]
                            wait_on_field_pos = True
                            begin = True

                        elif row[field_goal_result] == 'made' or row[field_goal_result] == 'missed' or row[field_goal_result] == 'blocked':
                            #print(line, row[field_goal_result], row[field_goal_result + 1])
                            distance = int(row[field_goal_result + 1])
                            if row[field_goal_result] == 'missed' or row[field_goal_result] == 'blocked':
                                distance *= -1
                                begin = True
                            writer.writerow([game_id, current_drive, field_pos, time_remaining, is_half_one, is_half_two, is_UTM, score_diff, p_timeout, d_timeout, 'fourth_down', int((int(field_pos) - int(row[field_position]))/10), int(field_pos) - int(row[field_position]), '1', distance, '0', 'NA', 'NA', 'NA', int((time_remaining - int(row[half_seconds]))/30)]) 
                            

                        line += 1
