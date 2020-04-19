import os
import time
from models import train_models
import numpy as np
import sys

def next_state(current_state, new_field_pos, runoff, score, keep_pos=False):
    #get rid of invalid states
    if new_field_pos <= 0:
        new_field_pos = 5
    if new_field_pos >= 100:
        new_field_pos = 95
    new_time = current_state[1] - runoff
    new_time = max(0, new_time)
    new_score = current_state[4] + score if  abs(current_state[4] + score) < 200 else current_state[4]
    if abs(current_state[4] + score) <= 100: 
        new_score = current_state[4] + score

    if not keep_pos:
        new_score *= -1

    return [int(new_field_pos), new_time, current_state[2], current_state[3], new_score]


def prob(current_state, win_probability, models):

    #print(current_state)
    if not current_state:
        #current state must be invalid
        return 0.0
    #models = [om, yd, fg, sm, pm, tr, to, fd]
    om = models[0]
    yd = models[1]
    fg = models[2]
    sm = models[3]
    pm = models[4]
    tr = models[5]
    to = models[6]
    fd = models[7]

    if win_probability[tuple(current_state)] > -1.0:
        return win_probability[tuple(current_state)]
    
    elif current_state[1] <= 0:
        #there is no time remaining
        if current_state[4] > 0:            
            win_probability[tuple(current_state)] = 1.0
        elif current_state[4] < 0:
            win_probability[tuple(current_state)] = 0.0
        else:
            win_probability[tuple(current_state)] = 0.5

    elif abs(current_state[4]) >= 100:
        #mercy rule
        win_probability[tuple(current_state)] = 1.0 if current_state[4] > 0 else 0.0

    else:
        #oh boy
        p_w = 0.0
        runoff_predictions = tr.predict(current_state)
        outcome_predictions = om.predict(current_state)
        #print(runoff_predictions)
        #0 = def_TD, 1= eoh, 2=4th, 3 = safety, 4 = TD, 5 = TO
        for i in range(0, len(tr.classes)):
            runoff = int(tr.classes[i] * 30 + 30)
            #don't consider if a 0% chance
            if runoff_predictions[i] == 0.0:
                continue

            #runoff means time expires
            if current_state[1] - runoff <= -30:
                p_w += runoff_predictions[i] * win_probability[(current_state[0], current_state[1] - 30, current_state[2], current_state[3], current_state[4])]
                continue


            pw_TD = 0.0
            #probability of TD is more than 0
            if outcome_predictions[4] > 0:
                pw_2P = 0.5 * (1 - prob(next_state(current_state, 75, runoff, 8), win_probability, models)) + 0.5 * (1 - prob(next_state(current_state, 75, runoff, 6), win_probability, models))
                pw_XP = 0.95 * (1 - prob(next_state(current_state,75, runoff, 7), win_probability, models)) + 0.05 * (1 - prob(next_state(current_state, 75, runoff, 6), win_probability, models))
                pw_TD = max(pw_2P, pw_XP)


            pw_TO = 0.0
            if outcome_predictions[5] > 0:
                turnover_predictions = to.predict(current_state)
                pw_TO = 0.0

                for j in range(0, len(to.classes)):
                    if turnover_predictions[j] > 0:
                        pw_TO += turnover_predictions[j] * (1 - prob(next_state(current_state, to.classes[j] * 5, runoff, 0), win_probability, models))

            pw_safety = 0.0
            if outcome_predictions[3] > 0:
                pw_safety = (1 - prob(next_state(current_state, 75, runoff, -2), win_probability, models))
            
            
            pw_eoh = 0.0
            if current_state[4] > 0:
                pw_eoh = 1.0
            elif current_state[4] < 0:
                pw_eoh = 0.0
            else:
                pw_eoh = 0.5
            
            
            pw_defTD = 0.0
            if outcome_predictions[0]:
                pw_def2P = 0.5 * prob(next_state(current_state, 75, runoff, -8, True), win_probability, models) + 0.5 * prob(next_state(current_state, 75, runoff, -6, True), win_probability, models)
                pw_defXP = 0.95 * prob(next_state(current_state,75, runoff, -7, True), win_probability, models) + 0.05 * prob(next_state(current_state, 75, runoff, -6, True), win_probability, models)
                pw_defTD = min(pw_def2P, pw_defXP)


            pw_4th = 0.0
            if outcome_predictions[2] > 0:
                yd_predictions = yd.predict(current_state)
                for j in range(0, len(yd.classes)):
                    if yd_predictions[j] > 0:
                        small_total = 0.0
                        predictions = sm.predict(yd.classes[j])
                        for k in range(0,10):
                            #if the probability is greater than 0
                            if predictions[k] > 0:
                                yards = yd.classes[j] + k
                                new_position = current_state[0] - yards
                                if new_position <= 0:
                                    new_position = 5
                                if new_position >= 100:
                                    new_position = 95
                               
                                new_state = current_state.copy()
                                new_state[0] = int(new_position)
                                if new_position <= 10 or current_state[0] <= 10:
                                    togo = new_position
                                elif current_state[0] > new_position:
                                    togo = 10 - ((current_state[0] - new_position) % 10)
                                else:
                                    togo = new_position - (current_state[0] - 10)
                                if togo != new_position:
                                    pw_GO = fd.predict(togo)[1] * prob(next_state(new_state, new_state[0] - togo, runoff, 0, True), win_probability, models) +  fd.predict(togo)[0] * (1 - prob(next_state(new_state, 100 - new_state[0], runoff, 0), win_probability, models))
                                else:
                                    pw_2P = 0.5 * (1 - prob(next_state(current_state, 75, runoff, 8), win_probability, models)) + 0.5 * (1 - prob(next_state(current_state, 75, runoff, 6), win_probability, models))
                                    pw_XP = 0.95 * (1 - prob(next_state(current_state,75, runoff, 7), win_probability, models)) + 0.05 * (1 - prob(next_state(current_state, 75, runoff, 6), win_probability, models))
                                    pw_GO = fd.predict(togo)[1] * max(pw_XP, pw_2P) + fd.predict(togo)[0] * (1 - prob(next_state(new_state, 100 - new_state[0], runoff, 0), win_probability, models))
                                    

                                pw_FG = fg.predict(new_position + 17) * (1 - prob(next_state(new_state, 75, runoff, 3), win_probability, models)) + (1 - fg.predict(new_position + 17)) * (1 - prob(next_state(new_state, 100 - new_state[0], runoff, 0), win_probability, models))
                                
                                punt_classes = pm.classes
                                pw_punt = 0.0
                                punt_predictions = pm.predict(new_state[0])
                                for p in range(0,len(pm.classes)):
                                    if punt_predictions[p] > 0:
                                        pw_punt += punt_predictions[p] * (1 - prob(next_state(new_state, punt_classes[p] * 5 + 5, runoff, 0), win_probability, models))
                                small_total += predictions[k] * max(pw_GO, pw_FG, pw_punt)

                        pw_4th += yd_predictions[j] * small_total


            print(current_state)
            print("D: %.3f, E: %.3f, F:%.3f, S: %.3f, TD: %.3f, TO: %.3f" % (pw_defTD, pw_eoh, pw_4th, pw_safety, pw_TD, pw_TO))
            temp = outcome_predictions[0] * pw_defTD + outcome_predictions[1] * pw_eoh + outcome_predictions[2] * pw_4th + outcome_predictions[3] * pw_safety * outcome_predictions[4] * pw_TD + outcome_predictions[5] * pw_TO
            p_w += runoff_predictions[i] * temp
            # print(temp, p_w)
    
        win_probability[tuple(current_state)] = p_w
    
    return win_probability[tuple(current_state)]

def calculate_win_probabilities(filename, epochs=1000, max_seconds=1800, load_file=False):
    if os.path.isfile(filename):
        if (load_file):
            print("loading data from %s" % filename)
            input_file = open(filename, 'rb')
            win_probability = np.load(input_file)
            input_file.close()
        else:
            print("%s already exists. Don't want to overwrite existing data" % filename)
            return None
    elif load_file:
        print("%s does not exist" % filename)
        return None
    else:
        win_probability = np.full((100,1801,2,2,201), -1.0)

    t = time.localtime()
    print("START: %02d/%02d/%02d: %02d:%02d:%02d" % (t.tm_mon, t.tm_mday, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))

    models = train_models(epochs)

    t = time.localtime()
    print("BEGINNING DP: %02d/%02d/%02d: %02d:%02d:%02d" % (t.tm_mon, t.tm_mday, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))
    state = [75,0,1,0,0]
    p = []
    for i in range(0,max_seconds + 1,30):
        state[1] = i
        p_w = prob(state, win_probability, models)
        print("%d seconds: P_w = %.3f" % (i, p_w))
        p.append(p_w)
    t = time.localtime()
    print("FINISHED DP: %02d/%02d/%02d: %02d:%02d:%02d" % (t.tm_mon, t.tm_mday, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))
    print("saving results...")
    outfile = open(filename, 'wb')
    np.save(outfile, win_probability, models)
    t = time.localtime()
    print("FINISH: %02d/%02d/%02d: %02d:%02d:%02d" % (t.tm_mon, t.tm_mday, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))
    print(p)
    outfile.close()
    
    #return models trained in order to prevent having to retrain them
    return models



if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] != '-l':
        _ = calculate_win_probabilities(sys.argv[1], epochs=1000, max_seconds=90)
    elif len(sys.argv) == 3 and sys.argv[1] == '-l':
        _ = calculate_win_probabilities(sys.argv[2], epochs=1000, max_seconds=1800, load_file=True)
    else:
        print("incorrect usage")
    while 1:
        pass






