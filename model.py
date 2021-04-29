import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, AlbertModel, DistilBertModel, RobertaModel, XLNetModel


model_ckpts = {'bert': "bert-base-uncased",
               'albert': "albert-base-v2",
               'roberta': "roberta-base",
               'xlnet': "xlnet-base-cased",
               'distilbert': "distilbert-base-uncased"}


class LMNet(nn.Module):
    def __init__(self,
                 finetuning=True,
                 lm='bert',
                 data_path=None,
                 use_literal_gnn=True,
                 use_digital_gnn=True,
                 use_structure_gnn=True,
                 use_name_gnn=True,
                 bert_path=None):
        super().__init__()

        self.path = data_path

        self.use_literal_gnn = use_literal_gnn
        self.use_digital_gnn = use_digital_gnn
        self.use_structure_gnn = use_structure_gnn
        self.use_name_gnn = use_name_gnn

        self.Literal_embeddingA = None
        self.Literal_embeddingB = None
        self.Digital_embeddingA = None
        self.Digital_embeddingB = None
        self.Structure_embeddingA = None
        self.Structure_embeddingB = None
        self.Name_embeddingA = None
        self.Name_embeddingB = None

        self.load_gnn_embedding()

        # load the model or model checkpoint
        if bert_path is None:
            if lm == 'bert':
                self.bert = BertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'distilbert':
                self.bert = DistilBertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'albert':
                self.bert = AlbertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'xlnet':
                self.bert = XLNetModel.from_pretrained(model_ckpts[lm])
            elif lm == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm])

        else:
            output_model_file = bert_path
            model_state_dict = torch.load(output_model_file,
                                          map_location=lambda storage, loc: storage)
            if lm == 'bert':
                self.bert = BertModel.from_pretrained(model_ckpts[lm],
                                                      state_dict=model_state_dict)
            elif lm == 'distilbert':
                self.bert = DistilBertModel.from_pretrained(model_ckpts[lm],
                                                            state_dict=model_state_dict)
            elif lm == 'albert':
                self.bert = AlbertModel.from_pretrained(model_ckpts[lm],
                                                        state_dict=model_state_dict)
            elif lm == 'xlnet':
                self.bert = XLNetModel.from_pretrained(model_ckpts[lm],
                                                       state_dict=model_state_dict)
            elif lm == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm],
                                                         state_dict=model_state_dict)

        self.finetuning = finetuning
        self.module_dict = nn.ModuleDict({})

        hidden_size = 768
        hidden_dropout_prob = 0.1

        vocab_size = 2

        self.module_dict['classification_dropout'] = nn.Dropout(hidden_dropout_prob)

        if self.use_literal_gnn or self.use_digital_gnn or self.use_digital_gnn or self.use_name_gnn:
            self.module_dict['classification_fc1'] = nn.Linear(hidden_size + 128 * 2, vocab_size)
        else:
            self.module_dict['classification_fc1'] = nn.Linear(hidden_size, vocab_size)

    def load_gnn_embedding(self):
        if self.use_literal_gnn:
            pathA = os.path.join(self.path, 'Literal_embeddingA.npy')
            pathB = os.path.join(self.path, 'Literal_embeddingB.npy')
            self.Literal_embeddingA = torch.tensor(np.load(pathA), requires_grad=False)
            self.Literal_embeddingB = torch.tensor(np.load(pathB), requires_grad=False)

        if self.use_digital_gnn:
            pathA = os.path.join(self.path, 'Digital_embeddingA.npy')
            pathB = os.path.join(self.path, 'Digital_embeddingB.npy')
            self.Digital_embeddingA = torch.tensor(np.load(pathA), requires_grad=False)
            self.Digital_embeddingB = torch.tensor(np.load(pathB), requires_grad=False)

        if self.use_structure_gnn:
            pathA = os.path.join(self.path, 'Structure_embeddingA.npy')
            pathB = os.path.join(self.path, 'Structure_embeddingB.npy')
            self.Structure_embeddingA = torch.tensor(np.load(pathA), requires_grad=False)
            self.Structure_embeddingB = torch.tensor(np.load(pathB), requires_grad=False)

        if self.use_name_gnn:
            pathA = os.path.join(self.path, 'Name_embeddingA.npy')
            pathB = os.path.join(self.path, 'Name_embeddingB.npy')
            self.Name_embeddingA = torch.tensor(np.load(pathA), requires_grad=False)
            self.Name_embeddingB = torch.tensor(np.load(pathB), requires_grad=False)

    def encode(self, x, batch_size=256):
        dropout = self.module_dict['classification_dropout']
        self.bert.eval()
        embedding = torch.zeros(x.size(0), 768)
        left = 0
        while left < x.size(0):
            right = min(left + batch_size, x.size(0))
            output = self.bert(x[left: right].cuda())
            pooler_output, _ = torch.max(output[0], dim=1)
            embedding[left: right] = dropout(pooler_output)
            left += batch_size
        return embedding

    def forward(self, x, sample, sentencesA=None, sentencesB=None):
        """Forward function of the models for classification."""

        dropout = self.module_dict['classification_dropout']
        fc1 = self.module_dict['classification_fc1']

        # Sentence features
        if self.training and self.finetuning:
            self.bert.train()
            output = self.bert(x)
            cls = output[0][:, 0, :]
            pairs = dropout(cls)
            output = self.bert(sentencesA)
            cls = output[0][:, 0, :]
            pairA = dropout(cls)
            output = self.bert(sentencesB)
            cls = output[0][:, 0, :]
            pairB = dropout(cls)

            pooled_output = pairs
        else:
            self.bert.eval()
            with torch.no_grad():
                output = self.bert(x)
                cls = output[0][:, 0, :]
                pairs = dropout(cls)

                pooled_output = pairs

        # Graph features
        embeddingA = torch.zeros(sample.size(0), 128, requires_grad=False).cuda().half()
        embeddingB = torch.zeros(sample.size(0), 128, requires_grad=False).cuda().half()
        if self.use_literal_gnn:
            Literal_embeddingA = torch.index_select(self.Literal_embeddingA, 0, sample[:, 0]).cuda().half()
            Literal_embeddingB = torch.index_select(self.Literal_embeddingB, 0, sample[:, 1]).cuda().half()
            embeddingA += Literal_embeddingA
            embeddingB += Literal_embeddingB
        if self.use_digital_gnn:
            Digital_embeddingA = torch.index_select(self.Digital_embeddingA, 0, sample[:, 0]).cuda().half()
            Digital_embeddingB = torch.index_select(self.Digital_embeddingB, 0, sample[:, 1]).cuda().half()
            embeddingA += Digital_embeddingA
            embeddingB += Digital_embeddingB
        if self.use_structure_gnn:
            Structure_embeddingA = torch.index_select(self.Structure_embeddingA, 0, sample[:, 0]).cuda().half()
            Structure_embeddingB = torch.index_select(self.Structure_embeddingB, 0, sample[:, 1]).cuda().half()
            embeddingA += Structure_embeddingA
            embeddingB += Structure_embeddingB
        if self.use_name_gnn:
            Name_embeddingA = torch.index_select(self.Name_embeddingA, 0, sample[:, 0]).cuda().half()
            Name_embeddingB = torch.index_select(self.Name_embeddingB, 0, sample[:, 1]).cuda().half()
            embeddingA += Name_embeddingA
            embeddingB += Name_embeddingB

        if self.use_literal_gnn or self.use_digital_gnn or self.use_digital_gnn or self.use_name_gnn:
            abs_embedding = torch.abs(embeddingA - embeddingB)
            dot_embedding = embeddingA * embeddingB

            pooled_output = torch.cat((pooled_output, abs_embedding, dot_embedding), dim=1)

        logits = fc1(pooled_output)

        y_hat = logits.argmax(-1)

        if self.training and self.finetuning:
            return logits, y_hat, pairA, pairB
        else:
            return logits, y_hat
