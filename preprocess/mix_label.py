
all_labels = {}
with open('labels/all_labels.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        x = line.strip().split(',')
        all_labels[x[0]] = x[1:]

print(all_labels)

with open('labels/shape/shape_anno_all.txt', 'r') as f:
    for line in f.readlines():
        label = line.strip().split(' ')
        
        if label[0] not in all_labels.keys():
            continue

        all_labels[label[0]].append(label[1:])

with open('labels/texture/fabric_ann.txt', 'r') as f:
    for line in f.readlines():
        label = line.strip().split(' ')
        
        if label[0] not in all_labels.keys():
            continue
        all_labels[label[0]].append(label[1:])

# write  'WOMEN-Tees_Tanks-id_00007838-03_2_side.jpg': [<temperature>, <sex>, <dress_code>, [<shapes>], [<textures>]] into csv
with open('labels/all_labels_aug.csv', 'w') as f:
    for key, value in all_labels.items():
        # flatten the value list
        flat_value = [item for sublist in value for item in sublist]
        f.write(f"{key},{','.join(flat_value)}\n")

