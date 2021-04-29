import argparse
import json
from utils import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Table to KG.',
        usage='table2kg_attr.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data_name', type=str, default='Structured/Fodors-Zagats')

    args = parser.parse_args(args)

    return args


if __name__ == '__main__':
    args = parse_args()

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[args.data_name]

    data_name = args.data_name

    path = config['path']

    tableA = os.path.join(path, 'tableA.csv')
    tableB = os.path.join(path, 'tableB.csv')

    tableA_name, tableB_name = data_name.split('/')[1].split('-')

    seeds_path = os.path.join(path, 'entity_seeds.txt')
    id2atts = os.path.join(path, 'id2atts.txt')
    id2entity_1 = os.path.join(path, 'id2entity_' + tableA_name + '.txt')
    id2entity_2 = os.path.join(path, 'id2entity_' + tableB_name + '.txt')
    id2relation_1 = os.path.join(path, 'id2relation_' + tableA_name + '.txt')
    id2relation_2 = os.path.join(path, 'id2relation_' + tableB_name + '.txt')
    atts_properties_1 = os.path.join(path, 'atts_properties_' + tableA_name + '.txt')
    atts_properties_2 = os.path.join(path, 'atts_properties_' + tableB_name + '.txt')
    triples_1 = os.path.join(path, 'triples_' + tableA_name + '.txt')
    triples_2 = os.path.join(path, 'triples_' + tableB_name + '.txt')

    input_path = os.path.join(path, 'seeds.csv')
    lines = list(csv.reader(open(input_path, 'r')))
    with open(seeds_path, 'w') as write_file:
        for j, line in enumerate(lines):
            if j != 0:
                if line[2] == '1':
                    write_file.write(line[0] + '\t' + line[1] + '\n')

    lines_A = list(csv.reader(open(tableA, 'r')))
    lines_B = list(csv.reader(open(tableB, 'r')))

    entityA = []
    entityB = []
    entityA2id = {}
    entityB2id = {}
    entityA_set = set()
    entityB_set = set()
    atts = lines_A[0]

    valuesA = set()
    valuesB = set()

    filter_list = ['phone', 'addr', 'id',
                   'price', 'ABV', 'class',
                   'Price', 'Time', 'Released']

    with open(atts_properties_1, 'w') as write_file:
        for i, line in enumerate(lines_A):
            if i != 0:
                if line[1].isspace() or len(line[1]) == 0:
                    line[1] = line[0]
                line[1] = line[1].strip()
                line[1] = line[1].replace('\t', ' ')
                if line[1] in entityA_set:
                    line[1] += '_' + line[0]
                entityA.append(line[1])
                entityA_set.add(line[1])
                for j in range(2, len(line)):
                    if len(line[j]) == 0 or line[j].isspace():
                        continue
                    line[j] = line[j].strip()
                    line[j] = line[j].replace('\t', ' ')
                    if line[j] in entityA_set:
                        line[j] += '_'
                        while line[j] in entityA_set:
                            line[j] += '_'
                    if atts[j] not in filter_list:
                        valuesA.add(line[j])
                    temp = line[1] + '\t' + atts[j] + '\t' + line[j] + '\n'
                    write_file.write(temp)

    with open(atts_properties_2, 'w') as write_file:
        for i, line in enumerate(lines_B):
            if i != 0:
                if line[1].isspace() or len(line[1]) == 0:
                    line[1] = line[0]
                line[1] = line[1].strip()
                line[1] = line[1].replace('\t', ' ')
                if line[1] in entityB_set:
                    line[1] += '_' + line[0]
                entityB.append(line[1])
                entityB_set.add(line[1])
                for j in range(2, len(line)):
                    if len(line[j]) == 0 or line[j].isspace():
                        continue
                    line[j] = line[j].strip()
                    line[j] = line[j].replace('\t', ' ')
                    if line[j] in entityA_set:
                        line[j] += '_'
                        while line[j] in entityA_set:
                            line[j] += '_'
                    if atts[j] not in filter_list:
                        valuesB.add(line[j])
                    temp = line[1] + '\t' + atts[j] + '\t' + line[j] + '\n'
                    write_file.write(temp)

    with open(id2atts, 'w') as write_file:
        for i, value in enumerate(atts):
            if i != 0:
                temp = str(i - 1) + '\t' + value + '\n'
                write_file.write(temp)

    entityA += list(valuesA)
    entityB += list(valuesB)

    with open(id2entity_1, 'w') as write_file:
        for i, value in enumerate(entityA):
            entityA2id[value] = str(i)
            temp = str(i) + '\t' + value + '\n'
            write_file.write(temp)

    with open(id2entity_2, 'w') as write_file:
        for i, value in enumerate(entityB):
            entityB2id[value] = str(i)
            temp = str(i) + '\t' + value + '\n'
            write_file.write(temp)

    relations = []
    relations2id = {}

    for item in atts:
        if item in filter_list:
            continue
        relations.append(item)

    with open(id2relation_1, 'w') as write_file_1, open(id2relation_2, 'w') as write_file_2:
        index = 0
        for i, item in enumerate(relations):
            temp = str(i) + '\t' + item + '\n'
            write_file_1.write(temp)
            write_file_2.write(temp)
            relations2id[item] = str(i)
            i += 1

    with open(triples_1, 'w') as write_file:
        for i, line in enumerate(lines_A):
            if i != 0:
                for j in range(1, len(line)):
                    if len(line[j]) == 0 or line[j].isspace() or atts[j] in filter_list:
                        continue
                    temp = str(i - 1) + '\t' + entityA2id[line[j]] + '\t' + relations2id[atts[j]] + '\n'
                    write_file.write(temp)

    with open(triples_2, 'w') as write_file:
        for i, line in enumerate(lines_B):
            if i != 0:
                for j in range(1, len(line)):
                    if len(line[j]) == 0 or line[j].isspace() or atts[j] in filter_list:
                        continue
                    temp = str(i - 1) + '\t' + entityB2id[line[j]] + '\t' + relations2id[atts[j]] + '\n'
                    write_file.write(temp)
