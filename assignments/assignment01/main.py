from ID3Tree import ID3Tree
import pandas as pd
import numpy as np

raw = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
])

data = pd.DataFrame(
    raw,
    columns = [
        'x1',
        'x2',
        'x3',
        'x4',
        'y'
    ]
)

features = data[data.columns[:-1]]
target   = data[data.columns[-1]]


tree = ID3Tree()

tree.train(
    features,
    target, 
    gain="entropy",
)

dump = tree.dump_nodes()
    
print(np.array(target))
print(tree.predict(features))

raw = np.array([
    ["S", "H", "H", "W", 0],
    ["S", "H", "H", "S", 0],
    ["O", "H", "H", "W", 1],
    ["R", "M", "H", "W", 1],
    ["R", "C", "N", "W", 1],
    ["R", "C", "N", "S", 0],
    ["O", "C", "N", "S", 1],
    ["S", "M", "H", "W", 0],
    ["S", "C", "N", "W", 1],
    ["R", "M", "N", "W", 1],
    ["S", "M", "N", "S", 1],
    ["O", "M", "H", "S", 1],
    ["O", "H", "N", "W", 1],
    ["R", "M", "H", "S", 0],
])

tennis = pd.DataFrame(
    raw,
    columns = [
        "O",
        "T",
        "H",
        "W",
        "y"
    ]
)

tennis_tree = ID3Tree()
tennis_tree.train(
    tennis[tennis.columns[:-1]],
    tennis[tennis.columns[-1]],
    gain="majority error",
)

print(np.array(tennis[tennis.columns[-1]]))
print(tennis_tree.predict(tennis[tennis.columns[:-1]]))