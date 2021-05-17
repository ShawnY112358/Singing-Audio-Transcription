import os

def shortest_note():
    dir = "../../ground_truth"
    with open('shortest_note.txt', 'w') as f:

        for list in os.listdir(dir):
            shortest = 1000
            with open(os.path.join(dir, list)) as fp:
                l = fp.read().split('\n')
                l = [float(i) for i in l]
                for i in range(int(len(l) / 3)):
                    time = l[i * 3 + 1] - l[i * 3]
                    if time < shortest:
                        shortest = time
                f.write(list.split('.')[0] + ' ' + str(shortest) + '\n')
                fp.close()


