from torchmetrics.text import CharErrorRate
import json

## change this line to the path to the json result of the test by Donut
f = open('donut/result/pthw2.json', 'r')

res = json.load(f)

# print(res['ground_truths'])



preds = [ t['text_sequence'] for t in res['predictions']]
target = [t['text_sequence'] for t in res['ground_truths']]

cer = CharErrorRate()
print(cer(preds, target))
