import json

# data set download from https://github.com/xinyadu/nqg/tree/master
# i just choose the train.json to do the experiment

if __name__ == '__main__':
    with open('input.txt', 'a', encoding='utf-8') as f1:
        with open('train.json','r') as f:
            data = json.load(f)
            for datalist in data:

                for dataitem in datalist['paragraphs']:
                    context = dataitem['context']
                    qas = dataitem['qas']
                    for qasitem in qas:
                        question = qasitem['question']
                        answer = qasitem['answers'][0]['text']
                        f1.write('[Q] '+question+'\n')
                        f1.write('[A] '+answer+'\n')