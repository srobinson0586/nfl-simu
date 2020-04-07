def next_state(current_state, new_field_pos, runoff, score, keep_pos=False):
    #get rid of invalid states
    if new_field_pos <= 0 or new_field_pos >= 100:
        return None
    new_time = current_state[1] - runoff
    if new_time <= -30:
        return None
    new_time = max(0, new_time)
    new_score = current_state[4] + score
    if not keep_pos:
        new_score *= -1

    return [int(new_field_pos), new_time, current_state[2], current_state[3], new_score]


def prob(current_state):

    if not current_state:
        #current state must be invalid
        return 0.0
    if win_probability[tuple(current_state)] > -1.0:
        return win_probability[tuple(current_state)]
    
    elif current_state[1] == 0:
        #there is no time remaining
        if current_state[4] > 0:            
            win_probability[tuple(current_state)] = 1.0
        elif current_state[4] < 0:
            win_probability[tuple(current_state)] = 0.0
        else:
            win_probability[tuple(current_state)] = 0.5

    else:
        #oh boy
        p_w = 0.0
        runoff_predictions = tr.predict(current_state)
        outcome_predictions = om.predict(current_state)
        #0 = def_TD, 1= eoh, 2=4th, 3 = safety, 4 = TD, 5 = TO
        for i in range(0, len(tr.classes)):
            runoff = int(tr.classes[i] * 30 + 30)
            if current_state[1] - runoff <= -30:
                continue
            
            pw_TD = 0.0
            if outcome_predictions[4] > 0:
                pw_2P = 0.5 * (1 - prob(next_state(current_state, 75, runoff, 8))) + 0.5 * (1- prob(next_state(current_state, 75, runoff, 6)))
                pw_XP = 0.95 * (1 - prob(next_state(current_state,75, runoff, 7))) + 0.05 * (1 - prob(next_state(current_state, 75, runoff, 6)))
                pw_TD = max(pw_2P, pw_XP)


            pw_TO = 0.0
            if outcome_predictions[5] > 0:
                turnover_predictions = to.predict(current_state)
                pw_TO = 0.0

                for j in range(0, len(to.classes)):
                    if turnover_predictions[j] > 0:
                        pw_TO += turnover_predictions[j] * (1 - prob(next_state(current_state, to.classes[j] * 5, runoff, 0)))

            pw_safety = 0.0
            if outcome_predictions[3] > 0:
                pw_safety = prob(next_state(current_state, 75, runoff, -2))
            
            
            pw_eoh = 0.0
            if outcome_predictions[1] > 0:
                temp_state = current_state.copy()
                temp_state[1] = 0
                pw_eoh = prob(temp_state)
            
            
            pw_defTD = 0.0
            if outcome_predictions[0]:
                pw_def2P = 0.5 * prob(next_state(current_state, 75, runoff, -8, True)) + 0.5 * prob(next_state(current_state, 75, runoff, -6, True))
                pw_defXP = 0.95 * prob(next_state(current_state,75, runoff, -7)) + 0.05 * prob(next_state(current_state, 75, runoff, -6, True))
                pw_defTD = min(pw_def2P, pw_defXP)


            pw_4th = 0.0
            if outcome_predictions[2] > 0:
                yd_classes = yd.classes
                yd_predictions = yd.predict(current_state)
                for j in range(0, len(yd.classes)):
                    if yd_predictions[j] > 0:
                        small_total = 0.0
                        predictions = sm.predict(yd_classes[j])
                        for k in range(0,9):
                            #if the probability is greater than 0
                            if predictions[k] > 0:
                                yards = j + k
                                new_position = current_state[0] + yards
                                new_state = current_state.copy()
                                new_state[0] = new_position
                                if new_position > 10:
                                    pw_GO = 0.5 * prob(next_state(new_state, new_state[0] - 10, runoff, 0, True)) + 0.5 * (1 - prob(next_state(new_state, 100 - new_state[0], runoff, 0)))
                                else:
                                    pw_GO = 0.5 * max(pw_XP, pw_2P) + 0.5 * (1 - prob(next_state(new_state, 100 - new_state[0], runoff, 0)))
                                    
                                pw_FG = fg.predict(new_position + 17) * (1 - prob(next_state(new_state, 75, runoff, 3))) + (1 - fg.predict(new_position + 17)) * (1 - prob(next_state(new_state, 100 - new_state[0], runoff, 0)))
                                
                                punt_classes = pm.classes
                                pw_punt = 0.0
                                punt_predictions = pm.predict(new_state[0])
                                for p in range(0,len(pm.classes)):
                                    if punt_predictions[p] > 0:
                                        pw_punt = punt_predictions[p] * (1 - prob(next_state(new_state, punt_classes[p] * 5 + 5, runoff, 0)))
                                small_total += predictions[k] * max(pw_GO, pw_FG, pw_punt)
                        pw_4th += yd_predictions[j] * small_total

            temp = outcome_predictions[0] * pw_defTD + outcome_predictions[1] * pw_eoh + outcome_predictions[2] * pw_4th + outcome_predictions[3] * pw_safety * outcome_predictions[4] * pw_TD + outcome_predictions[5] * pw_TO
            p_w += runoff_predictions[i] * temp
    
        win_probability[tuple(current_state)] = p_w
    
    return win_probability[tuple(current_state)]