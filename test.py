abc = [1, 2, 3, 4, 5, 11, 7, 8, 12, 9]
abc_new = []
idx_to_skip = [5, 8]

for i, item in enumerate(abc):
    if i not in idx_to_skip:
        abc_new.append(item)


stop = "here"