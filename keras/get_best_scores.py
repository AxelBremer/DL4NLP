import os 
import json

joe = os.walk('runs')
next(joe)

for x in joe:
    try:
        if(x[2][2] == 'history.json'):
            print(x[0])
            with open(x[0] + '\\' + x[2][2], 'r') as f:
                hist = json.load(f)
                print(min(list(hist['val_loss'])))
                print(max(list(hist['val_acc'])))
    except:
        pass