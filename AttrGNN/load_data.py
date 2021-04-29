import re
import json
import torch
import random
import numpy as np
from datetime import datetime
from utils import print_time_info
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


class BERT(object):
    # For entity alignment, the best layer is 1
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.model = BertModel.from_pretrained('./lm_model/bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.cls_token_id = self.tokenizer.encode(self.tokenizer.cls_token)[0]
        self.sep_token_id = self.tokenizer.encode(self.tokenizer.sep_token)[0]
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)

    def pooled_encode_batched(self, sentences, batch_size=512, layer=1, save_gpu_memory=False):
        # Split the sentences into batches and further encode
        sent_batch = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        outputs = []
        for batch in tqdm(sent_batch):
            out = self.pooled_bert_encode(batch, layer)
            if save_gpu_memory:
                out = out.cpu()
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def pooled_bert_encode(self, sentences, layer=1):
        required_layer_hidden_state, sent_lens = self.bert_encode(sentences, layer)
        required_layer_hidden_state = minus_mask(required_layer_hidden_state, sent_lens.to(self.device))
        # Max pooling
        required_layer_hidden_state, indices = torch.max(required_layer_hidden_state, dim=1, keepdim=False)
        return required_layer_hidden_state

    def bert_encode(self, sentences, layer=1):
        # layer: output the max pooling over the designated layer hidden state

        # Limit batch size to avoid exceed gpu memory limitation
        sent_num = len(sentences)
        assert sent_num <= 512

        # The 382 is to avoid exceed bert's maximum seq_len and to save memory
        sentences = [[self.cls_token_id] + self.tokenizer.encode(sent)[:382] + [self.sep_token_id] for sent in
                     sentences]
        sent_lens = [len(sent) for sent in sentences]
        max_len = max(sent_lens)
        sent_lens = torch.tensor(sent_lens)
        sentences = torch.tensor([sent + (max_len - len(sent)) * [self.pad_token_id] for sent in sentences]).to(
            self.device)
        with torch.no_grad():
            output = self.model(sentences)
            last_hidden_state = output.last_hidden_state
            pooled_output = output.pooler_output
            all_hidden_state = output.hidden_states
        assert len(all_hidden_state) == 13
        required_layer_hidden_state = all_hidden_state[layer]
        return required_layer_hidden_state, sent_lens


def minus_mask(inputs, input_lens):
    # Inputs shape = (batch_size, sent_len, embed_dim)
    # input_len shape = [batch_sie]
    # max_len scalar
    assert inputs.shape[0] == input_lens.shape[0]
    assert len(input_lens.shape) == 1
    assert len(inputs.shape) == 3
    device = inputs.device

    max_len = torch.max(input_lens)
    batch_size = inputs.shape[0]
    mask = torch.arange(max_len).expand(batch_size, max_len).to(device)
    mask = mask >= input_lens.view(-1, 1)
    mask = mask.float()
    mask = mask.reshape(-1, max_len, 1) * (-1e30)
    # Employ mask
    inputs = inputs + mask
    return inputs


def read_mapping(path):
    def _parser(lines):
        for idx, line in enumerate(lines):
            i, name = line.strip().split('\t')
            lines[idx] = (int(i), name)
        return dict(lines)

    return read_file(path, _parser)


def read_triples(path):
    """
    triple pattern: (head_id, tail_id, relation_id)
    """
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_seeds(path):
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_file(path, parse_func):
    num = -1
    with open(path, 'r', encoding='utf8') as f:
        line = f.readline().strip()
        if line.isdigit():
            num = int(line)
        else:
            f.seek(0)
        lines = f.readlines()

    lines = parse_func(lines)

    if len(lines) != num and num >= 0:
        print_time_info('File: %s has corruptted, data_num: %d/%d.' %
                        (path, num, len(lines)))
        raise ValueError()
    return lines


def _load_language(directory, language):
    triples = read_triples(directory / ('triples_' + language + '.txt'))
    id2entity = read_mapping(directory / ('id2entity_' + language + '.txt'))
    id2relation = read_mapping(directory / ('id2relation_' + language + '.txt'))
    return triples, id2entity, id2relation


def _load_seeds(directory, train_seeds_ratio):
    entity_seeds = read_seeds(directory / 'entity_seeds.txt')

    return entity_seeds


def _load_dbpedia_properties(data_path, entity2id):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    att_triples = []
    for line in lines:
        subject, property, value = line

        try:
            value = value.encode('utf8').decode('unicode_escape')
        except UnicodeDecodeError:
            pass
        ent_id = entity2id[subject]
        att = property
        att_triples.append((ent_id, value, att))
    # For all the triples: (head, tail, relation)
    return att_triples


def _get_train_value_and_attribute(train_ent_ids, att_triples):
    train_value_and_attribute = []
    train_ent_ids = set(train_ent_ids)
    for ent_id, value, att in att_triples:
        if ent_id in train_ent_ids:
            train_value_and_attribute.append((value, att))
    return train_value_and_attribute


def _split_digit_attribute_and_literal_attribute(value_and_attribute_pairs, digit_threshold, att_set):
    numeral = Numeral()
    att_is_number = {}
    for value, att_id in value_and_attribute_pairs:
        # 0 digit, 1 literal
        is_numeral, number = numeral.is_numeral(value)
        if att_id in att_is_number:
            if is_numeral:
                att_is_number[att_id][0] += 1
            else:
                att_is_number[att_id][1] += 1
        else:
            if is_numeral:
                att_is_number[att_id] = [1, 0]
            else:
                att_is_number[att_id] = [0, 1]

    digit_atts = {att for att, count in att_is_number.items() if count[0] / sum(count) > digit_threshold}
    literal_atts = {att for att in att_set if att not in digit_atts}
    digit_atts = list(digit_atts)
    digit_atts.sort()
    literal_atts = list(literal_atts)
    literal_atts.sort()
    digit_att2id = {digit_att: idx for idx, digit_att in enumerate(digit_atts)}
    literal_att2id = {literal_att: idx for idx, literal_att in enumerate(literal_atts)}
    return digit_att2id, literal_att2id


def _split_digit_and_literal_triple(att_triples, digit_att2id, literal_att2id):
    digit_triples = []
    literal_triples = []
    digit_num = 0
    literal_num = 0
    for ent_id, value, att in att_triples:
        if att in digit_att2id:
            digit_triples.append((ent_id, value, digit_att2id[att]))
            digit_num += 1
        else:
            literal_triples.append((ent_id, value, literal_att2id[att]))
            literal_num += 1
    return digit_triples, literal_triples


def get_cache_file_path(temp_file_dir, attribute_channel_name):
    if not temp_file_dir.exists():
        temp_file_dir.mkdir()
    assert attribute_channel_name in {'Literal', 'Digit', 'Attribute'}
    embedding_file_name = 'value_embedding'
    id2values_file_name = 'id2value'
    embedding_file_name = '%s_%s' % (embedding_file_name, attribute_channel_name)
    id2values_file_name = '%s_%s' % (id2values_file_name, attribute_channel_name)
    embedding_file_path = temp_file_dir / ('%s.npy' % embedding_file_name)
    id2values_file_path = temp_file_dir / ('%s.json' % id2values_file_name)
    return embedding_file_path, id2values_file_path


class ValueEmbedding(object):
    def __init__(self, device):
        self.bert = BERT()
        self.bert.to(device)

    def encode_value(self, value_seqs):
        value2id = {}
        for value_seq in value_seqs:
            for value in value_seq:
                if value not in value2id:
                    value2id[value] = len(value2id)
        # Add the [PAD] token for value embeddings
        value2id[self.bert.tokenizer.pad_token] = len(value2id)

        # id2value is a sequence of English text
        id2value = sorted(value2id.items(), key=lambda x: x[1])
        id2value = [item[0] for item in id2value]  # it is a list
        best_layer = 1
        value_embedding = self.bert.pooled_encode_batched(id2value, layer=best_layer, batch_size=128,
                                                          save_gpu_memory=True)
        value_embedding = value_embedding.numpy()
        return value_embedding, id2value

    def load_value(self, value_seqs, value_embedding_cache_path, id2value_cache_path):
        if value_embedding_cache_path.exists() and id2value_cache_path.exists():
            value_embedding = np.load(value_embedding_cache_path)
            with open(id2value_cache_path, 'r', encoding='utf8', errors='ignore') as f:
                id2value = json.load(f)
            print_time_info("Loaded value embedding from %s." % value_embedding_cache_path)
            print_time_info("Loaded values from %s." % id2value_cache_path)
        else:
            value_embedding, id2value = self.encode_value(value_seqs)
            np.save(value_embedding_cache_path, value_embedding)
            with open(id2value_cache_path, 'w', encoding='utf8', errors='ignore') as f:
                json.dump(id2value, f, ensure_ascii=False)
        assert len(value_embedding) == len(id2value)
        return value_embedding, id2value


class LoadData(object):
    def __init__(self, train_seeds_ratio, directory, nega_sample_num, name_channel,
                 attribute_channel, digit_literal_channel, device='cpu'):
        self.device = device
        self.directory = directory
        self.nega_sample_num = nega_sample_num
        self.train_seeds_ratio = train_seeds_ratio
        self.language_sr, self.language_tg = directory.name.split('-')
        self.load_seed_alignment()
        self.load_structure_feature()
        if name_channel:
            self.load_name_feature()
        if attribute_channel or digit_literal_channel:
            self.load_attribute_feature(attribute_channel, digit_literal_channel)
        self.negative_sample()

    def update_negative_sample(self, sr_nega_sample, tg_nega_sample):
        # negative sample shape = (data_len, negative_sample_num)
        assert sr_nega_sample.shape == (len(self.train_sr_ent_seeds_ori), self.nega_sample_num)
        assert tg_nega_sample.shape == (len(self.train_tg_ent_seeds_ori), self.nega_sample_num)

        if not (hasattr(self, "sr_posi_sample") and hasattr(self, "tg_posi_sample")):
            sr_posi_sample = np.tile(self.train_sr_ent_seeds_ori.reshape((-1, 1)), (1, self.nega_sample_num))
            tg_posi_sample = np.tile(self.train_tg_ent_seeds_ori.reshape((-1, 1)), (1, self.nega_sample_num))
            self.sr_posi_sample = torch.from_numpy(sr_posi_sample.reshape((-1, 1))).to(self.device)
            self.tg_posi_sample = torch.from_numpy(tg_posi_sample.reshape((-1, 1))).to(self.device)

        sr_nega_sample = sr_nega_sample.reshape((-1, 1))
        tg_nega_sample = tg_nega_sample.reshape((-1, 1))
        sr_nega_sample = sr_nega_sample.cuda()
        tg_nega_sample = tg_nega_sample.cuda()
        self.train_sr_ent_seeds = torch.cat((self.sr_posi_sample, sr_nega_sample), dim=1)
        self.train_tg_ent_seeds = torch.cat((self.tg_posi_sample, tg_nega_sample), dim=1)

    def negative_sample(self):
        # Randomly negative sample
        sr_nega_sample = negative_sample(self.train_sr_ent_seeds_ori, self.sr_ent_num, self.nega_sample_num)
        tg_nega_sample = negative_sample(self.train_tg_ent_seeds_ori, self.tg_ent_num, self.nega_sample_num)


        sr_nega_sample = torch.from_numpy(sr_nega_sample).to(self.device)
        tg_nega_sample = torch.from_numpy(tg_nega_sample).to(self.device)

        self.update_negative_sample(sr_nega_sample, tg_nega_sample)

    def load_structure_feature(self):
        # Load triples and entity mapping
        id2atts = read_mapping(self.directory / 'id2atts.txt')
        self.att2id = {att: idx for idx, att in id2atts.items()}
        self.att_num = len(self.att2id)
        self.triples_sr, self.id2entity_sr, self.id2relation_sr = _load_language(self.directory, self.language_sr)
        self.triples_tg, self.id2entity_tg, self.id2relation_tg = _load_language(self.directory, self.language_tg)
        self.sr_ent_num = len(self.id2entity_sr)
        self.tg_ent_num = len(self.id2entity_tg)

    def load_name_feature(self):
        # Load translations

        id2entity_sr = sorted(self.id2entity_sr.items(), key=lambda x: x[0])
        sr_text = [x[1] for x in id2entity_sr]
        id2entity_tg = sorted(self.id2entity_tg.items(), key=lambda x: x[0])
        tg_text = [x[1] for x in id2entity_tg]
        bert = BERT()
        bert.to(self.device)
        self.sr_embed = bert.pooled_encode_batched(sr_text, layer=1)
        self.tg_embed = bert.pooled_encode_batched(tg_text, layer=1)
        del bert

    def load_seed_alignment(self):
        # Load alignment seeds
        entity_seeds = _load_seeds(self.directory, self.train_seeds_ratio)

        self.entity_seeds = entity_seeds  # The entity seeds in the original order

        # train_ent_seeds shape = [length, 2]
        train_sr_ent_seeds_ori, train_tg_ent_seeds_ori = zip(*entity_seeds)

        self.train_sr_ent_seeds_ori = np.asarray(train_sr_ent_seeds_ori)
        self.train_tg_ent_seeds_ori = np.asarray(train_tg_ent_seeds_ori)

    def load_attribute_feature(self, load_attribute, load_digit_literal):
        directory = self.directory
        language_sr = self.language_sr
        language_tg = self.language_tg

        entity2id_sr = {}
        entity2id_tg = {}
        for key, value in self.id2entity_sr.items():
            entity2id_sr[value] = key
        for key, value in self.id2entity_tg.items():
            entity2id_tg[value] = key

        att_triples_sr = _load_dbpedia_properties(directory / ("atts_properties_%s.txt" % language_sr),
                                                  entity2id_sr)
        att_triples_tg = _load_dbpedia_properties(directory / ("atts_properties_%s.txt" % language_tg),
                                                  entity2id_tg)

        temp_file_dir = directory / 'running_temp'
        value_embed_encoder = ValueEmbedding(self.device)

        if load_attribute:
            print('load attribute...')
            self.att_triples_sr = [(ent_id, value, self.att2id[att]) for ent_id, value, att in att_triples_sr]
            self.att_triples_tg = [(ent_id, value, self.att2id[att]) for ent_id, value, att in att_triples_tg]

            ent_id_seq_sr, att_id_seq_sr, value_seq_sr = transform_triple2seq(self.att_triples_sr, language_sr)
            ent_id_seq_tg, att_id_seq_tg, value_seq_tg = transform_triple2seq(self.att_triples_tg, language_tg)
            value_embed_cache_path, id2value_cache_path = get_cache_file_path(temp_file_dir, 'Attribute')
            self.value_embedding, self.id2value = value_embed_encoder.load_value(value_seq_sr + value_seq_tg,
                                                                                 value_embed_cache_path,
                                                                                 id2value_cache_path, )
            value2id = {value: idx for idx, value in enumerate(self.id2value)}
            value_id_seq_sr = [[value2id.get(value, value2id['[PAD]']) for value in value_seq] for value_seq in
                               value_seq_sr]
            value_id_seq_tg = [[value2id.get(value, value2id['[PAD]']) for value in value_seq] for value_seq in
                               value_seq_tg]

            attribute_triples_sr = []
            for ent_id, att_seq, val_seq in zip(ent_id_seq_sr, att_id_seq_sr, value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    attribute_triples_sr.append((ent_id, val, att))
            self.attribute_triples_sr = torch.tensor(attribute_triples_sr)

            attribute_triples_tg = []
            for ent_id, att_seq, val_seq in zip(ent_id_seq_tg, att_id_seq_tg, value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    attribute_triples_tg.append((ent_id, val, att))
            self.attribute_triples_tg = torch.tensor(attribute_triples_tg)

        if load_digit_literal:
            train_value_and_attribute_sr = _get_train_value_and_attribute(self.train_sr_ent_seeds_ori, att_triples_sr)
            train_value_and_attribute_tg = _get_train_value_and_attribute(self.train_tg_ent_seeds_ori, att_triples_tg)

            digit_threshold = 0.5
            digit_att2id, literal_att2id = _split_digit_attribute_and_literal_attribute(
                train_value_and_attribute_sr + train_value_and_attribute_tg, digit_threshold, set(self.att2id.keys()))
            self.digit_att2id = digit_att2id
            self.literal_att2id = literal_att2id
            self.digit_att_num = len(digit_att2id)
            self.literal_att_num = len(literal_att2id)
            digit_triples_sr, literal_triples_sr = _split_digit_and_literal_triple(att_triples_sr, digit_att2id,
                                                                                   literal_att2id)
            digit_triples_tg, literal_triples_tg = _split_digit_and_literal_triple(att_triples_tg, digit_att2id,
                                                                                   literal_att2id)

            digit_ent_id_seq_sr, digit_att_id_seq_sr, digit_value_seq_sr = transform_triple2seq(digit_triples_sr,
                                                                                                language_sr, False)
            digit_ent_id_seq_tg, digit_att_id_seq_tg, digit_value_seq_tg = transform_triple2seq(digit_triples_tg,
                                                                                                language_tg, False)

            literal_ent_id_seq_sr, literal_att_id_seq_sr, literal_value_seq_sr = transform_triple2seq(
                literal_triples_sr, language_sr, False)
            literal_ent_id_seq_tg, literal_att_id_seq_tg, literal_value_seq_tg = transform_triple2seq(
                literal_triples_tg, language_tg, False)
            literal_value_embed_cache_path, literal_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Literal')
            digit_value_embed_cache_path, digit_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Digit')

            self.literal_value_embedding, self.literal_id2value = value_embed_encoder.load_value(
                literal_value_seq_sr + literal_value_seq_tg, literal_value_embed_cache_path,
                literal_id2value_cache_path)

            self.digit_value_embedding, self.digit_id2value = value_embed_encoder.load_value(
                digit_value_seq_sr + digit_value_seq_tg, digit_value_embed_cache_path, digit_id2value_cache_path, )

            literal_value2id = {value: idx for idx, value in enumerate(self.literal_id2value)}
            digit_value2id = {value: idx for idx, value in enumerate(self.digit_id2value)}

            digit_value_id_seq_sr = [[digit_value2id.get(value, digit_value2id['[PAD]']) for value in value_seq] for
                                     value_seq in digit_value_seq_sr]
            digit_value_id_seq_tg = [[digit_value2id.get(value, digit_value2id['[PAD]']) for value in value_seq] for
                                     value_seq in digit_value_seq_tg]

            literal_value_id_seq_sr = [[literal_value2id.get(value, literal_value2id['[PAD]']) for value in value_seq]
                                       for value_seq in literal_value_seq_sr]
            literal_value_id_seq_tg = [[literal_value2id.get(value, literal_value2id['[PAD]']) for value in value_seq]
                                       for value_seq in literal_value_seq_tg]

            literal_triples_sr = []
            for ent_id, att_seq, val_seq in zip(literal_ent_id_seq_sr, literal_att_id_seq_sr, literal_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    literal_triples_sr.append((ent_id, val, att))
            self.literal_triples_sr = torch.tensor(literal_triples_sr)

            literal_triples_tg = []
            for ent_id, att_seq, val_seq in zip(literal_ent_id_seq_tg, literal_att_id_seq_tg, literal_value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    literal_triples_tg.append((ent_id, val, att))
            self.literal_triples_tg = torch.tensor(literal_triples_tg)

            digital_triples_sr = []
            for ent_id, att_seq, val_seq in zip(digit_ent_id_seq_sr, digit_att_id_seq_sr, digit_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    digital_triples_sr.append((ent_id, val, att))
            self.digital_triples_sr = torch.tensor(digital_triples_sr)

            digital_triples_tg = []
            for ent_id, att_seq, val_seq in zip(digit_ent_id_seq_tg, digit_att_id_seq_tg, digit_value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    digital_triples_tg.append((ent_id, val, att))
            self.digital_triples_tg = torch.tensor(digital_triples_tg)

        del value_embed_encoder


def negative_sample(pos_ids, data_range, nega_sample_num):
    # Output shape = (data_len, negative_sample_num)
    nega_ids_arrays = np.random.randint(low=0, high=data_range - 1, size=(len(pos_ids), nega_sample_num))
    for idx, pos_id in enumerate(pos_ids):
        for j in range(nega_sample_num):
            if nega_ids_arrays[idx][j] >= pos_id:
                nega_ids_arrays[idx][j] += 1
    assert nega_ids_arrays.shape == (len(pos_ids), nega_sample_num), print(nega_ids_arrays.shape)
    return nega_ids_arrays


def transform_triple2seq(att_triples, language, concate_values=False):
    # ent_id_seq = [ent1_id, ent2_id, ent3_id...]
    # prop_num = [ent1_num_prop, ent2_num_prop...]
    # att_id_seq = [[ent1_prop1_id, ent1_prop2_id, ...]...]
    # value_seq = [[ent1_value1, ent1_value2, ...]...]
    # Fixme: select the first 20 attributes
    # Fixme: Original average property number 26 --> only one property average property number 15.9 --> top 20 property 10.09

    top_k_att = 3
    ent_id_seq = []
    prop2value_seq = []
    for ent_id, value, att_id in att_triples:
        if len(ent_id_seq) == 0:
            ent_id_seq.append(ent_id)
            prop2value_seq.append(dict())
        if ent_id != ent_id_seq[-1]:
            ent_id_seq.append(ent_id)
            prop2value_seq.append(dict())
        if not concate_values:
            prop2value_seq[-1][att_id] = value
        else:
            if att_id in prop2value_seq[-1]:
                prop2value_seq[-1][att_id] += '. ' + value
            else:
                prop2value_seq[-1][att_id] = value
    att_id_seq = []
    value_seq = []
    for prop2value in prop2value_seq:
        att_ids, values = zip(*list(prop2value.items()))
        assert len(values) == len(att_ids)
        att_id_seq.append(att_ids[:top_k_att])
        value_seq.append(values[:top_k_att])
    return ent_id_seq, att_id_seq, value_seq


def construct_ent_id2info(ent_num, ent_id_seq, att_id_seq, value_id_seq, att_pad_id, value_pad_id, language):
    top_k_att = 3

    assert len(ent_id_seq) == len(att_id_seq) == len(value_id_seq)
    entid2atts = [[] for _ in range(ent_num)]
    entid2values = [[] for _ in range(ent_num)]
    for ent_id, att_ids, value_ids in zip(ent_id_seq, att_id_seq, value_id_seq):
        entid2atts[ent_id] += att_ids
        entid2values[ent_id] += value_ids

    entid2atts = [item[:top_k_att] for item in entid2atts]
    entid2values = [item[:top_k_att] for item in entid2values]

    max_len1 = max(len(item) for item in entid2atts)
    max_len2 = max(len(item) for item in entid2values)
    assert max_len1 == max_len2 == top_k_att
    ent2att_num = np.zeros(ent_num, dtype=np.int)
    ent2atts = np.ones((ent_num, max_len1), dtype=np.int) * att_pad_id
    ent2values = np.ones((ent_num, max_len1), dtype=np.int) * value_pad_id

    att_num = 0
    for idx, (atts, values) in enumerate(zip(entid2atts, entid2values)):
        assert len(atts) == len(values)
        ent2att_num[idx] = len(atts)
        ent2atts[idx, :len(atts)] = atts
        ent2values[idx, :len(atts)] = values
        att_num += len(atts)
    return ent2att_num, ent2atts, ent2values


class Numeral(object):
    def __init__(self):
        self.regex = {'year': re.compile(r'^\d{3,4}$'),
                      'date': re.compile(r'^(\d+)-(\d+)-(\d+)$'),
                      'month_day': re.compile(r'^--(\d{2})-(\d{2})$'),
                      'ABV_price': re.compile(r'(\d+)'),
                      'time': re.compile(r'(\d+):(\d+)')}
        self.regex_func = {'year': lambda x: (int(x.group(0)), None, None),
                           'date': lambda x: (int(x.group(1)), int(x.group(2)), int(x.group(3))),
                           'month_day': lambda x: (None, int(x.group(1)), int(x.group(2))),
                           'ABV_price': lambda x: (int(x.group(0))),
                           'time': lambda x: (int(x.group(1)), int(x.group(2)))}

    def is_numeral(self, text):
        is_numeral, result = self.__float_pattern(text)
        if is_numeral:
            return is_numeral, result
        else:
            patterns = ['ABV_price', 'year', 'date', 'month_day', 'time']
            for pattern in patterns:
                is_numeral, result = self.__regex_pattern(text, pattern)
                if is_numeral:
                    return is_numeral, result
            return False, None

    def __regex_pattern(self, text, regex_name):
        regex = self.regex[regex_name]
        result = regex.match(text)
        if result:
            return True, self.regex_func[regex_name](result)
        return False, None

    def __month_year_pattern(self, text):
        try:
            data = datetime.strptime(text, '%B %Y')
            return True, (data.year, data.month, None)
        except ValueError:
            return False, None
        except:
            raise Exception()

    def __float_pattern(self, text):
        special_patterns = ['inf', 'nan']
        for pattern in special_patterns:
            if text.lower().find(pattern) >= 0:
                return False, None
        try:
            data = float(text)
            return True, data
        except ValueError:
            return False, None
        except:
            raise Exception()
