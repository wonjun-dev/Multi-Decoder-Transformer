from utils.data_utils import smi_tokenizer


def write_data(data, path, dtype):
    if dtype == 'txt':
        for idx, row in enumerate(data):
            if idx == 0:
                with open(path, 'w') as f:
                    f.write(row + '\n')
            else:
                with open(path, 'a') as f:
                    f.write(row + '\n')
    elif dtype == 'np':
        np.save(path, data)

with open('data/chembl_lm/ChemBL-LM_train.csv', 'r') as f:
    lines = f.readlines()

lines = lines[1:]
lines = [smi_tokenizer(line.split(',')[0]) for line in lines]

write_data(lines, 'data/chembl_lm/src-train.txt', 'txt')
write_data(lines, 'data/chembl_lm/tgt-train.txt', 'txt')

print(lines[:10])

with open('data/chembl_lm/ChemBL-LM_val.csv', 'r') as f:
    lines = f.readlines()

lines = lines[1:]
lines = [smi_tokenizer(line.split(',')[0]) for line in lines]

write_data(lines, 'data/chembl_lm/src-val.txt', 'txt')
write_data(lines, 'data/chembl_lm/tgt-val.txt', 'txt')

print(lines[:10])
