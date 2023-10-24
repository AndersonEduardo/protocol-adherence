import spacy
import nltk
import Levenshtein
import torch


import numpy as np
import pylightxl as xl

from nltk.stem import RSLPStemmer

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from sentence_transformers import SentenceTransformer, util

from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads


from parameters import *


class StatisticalFramework:

    def __init__(self, parameters_filepath:str, rule_parameter:str) -> None:

        super().__init__()
        self.__filepath = self.set_filepath(parameters_filepath)
        self.datatypes = self.get_datatypes()
        self.actions, self.conditions, self.rules, self.config = self.load_xlsx()
        self.syntax_elements = SYNTAX_ELEMENTS
        self.rule_parameter = rule_parameter
        self.nlp = spacy.load("pt_core_news_lg")
        self.nltk_stemmer = RSLPStemmer().stem
        self.X, self.Y = self.set_reference_data()
        # self.null_distribution_data = self.set_null_distribution(n_iterations = 5000)

        self.bow, self.tfidf_vectorizer = self.set_bow() 
        self.bow_models =  self.get_models(self.bow)

        self.bert_model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)

        self.sentence_transformer_model = SentenceTransformer('neuralmind/bert-large-portuguese-cased')

    #############################
    ### SECTION: LOADING DATA ###
    #############################


    def set_filepath(self, filepath:str) -> str:
        # TODO: implementar tratamento de excecoes completo

        if not isinstance(filepath, str):

            raise ValueError('`filepath` must be a python str.')

        return filepath


    def set_datatype(self, parameter, value:list) -> list:

        if not isinstance(parameter, str) and \
           not isinstance(value, list):

            raise TypeError('`parameter` must be a python str, and `value` must be a python list.')

        datatype = self.datatypes.get(parameter)

        if datatype[0] == int:

            return [int(float(v)) for v in value]
        
        elif datatype[0] == float:

            return [float(v) for v in value]

        elif datatype[0] == str:

            return [str(v) for v in value]

        else:

            raise TypeError('The only supported datatypes are: int, float and str.')


    def get_datatypes(self):

        if isinstance(DATATYPE_DICT, dict):

            return DATATYPE_DICT

        else:

            raise TypeError("`DATATYPE_DICT` must be a python dict.")


    def load_xlsx(self) -> None:

        actions = list()
        conditions = set()
        rules = list()
        config = None

        xlsx_file = xl.readxl(fn=self.__filepath)
        sheet_names = xlsx_file.ws_names

        for sheet_name in sheet_names:

            if sheet_name.strip().lower() != 'config':

                # updating the set of decision table actions
                actions.append(sheet_name.strip().upper())

            sheet_data = xlsx_file.ws(ws=sheet_name)
            item = dict()

            for i in range(1, len(sheet_data.row(row=1)) + 1):

                column_data = sheet_data.col(col=i)
                col_name = str(column_data[0]).strip().upper()
                col_contents = [
                    str(x).strip().lower()
                    for x in column_data[1:]
                    if not str(x).isspace() and not x == ''
                ]

                if (not col_name.isspace() and 
                    not col_name=='' and 
                    sheet_name.strip().lower() != 'config'):

                    # updating the set of decision table conditions
                    conditions.update({col_name})

                    item[col_name] = self.set_datatype(col_name, col_contents)

                elif sheet_name.strip().lower() == 'config':

                    item[col_name] = col_contents

                else:

                    pass

            if sheet_name.strip().lower() != 'config':

                # updating the set of decision table rules
                rules.append(item)

            else:

                # updating the set of decision table config
                config = item


        return actions, conditions, rules, config


    def set_reference_data(self):

        X = list()
        Y = list()

        for (rule, action) in zip(self.rules, self.actions):

            parameter_contents = rule.get(self.rule_parameter)

            for content in parameter_contents:

                doc = self.nlp(content)
                x_i = set()

                for token in doc:

                    if token.pos_.strip().upper() in self.syntax_elements:

                        # usando lematização
                        # x_i.update({token.lemma_.lower()})

                        # usando stematização
                        x_i.update({ self.nltk_stemmer(token.text.strip().lower()) })

                X.append(' '.join(x_i))
                Y.append(action)
        
        return X, Y


    def set_null_distribution(self, n_iterations:int):
        
        null_distribution_data_raw = self.get_null_distribution(n_iterations=n_iterations)

        null_distribution_data = pd.DataFrame(
            null_distribution_data_raw,
            columns=['protocol', 'score']
        )

        null_distribution_data = null_distribution_data.dropna()

        null_distribution_data['score'] = null_distribution_data['score'].apply(lambda x: round(x, 3))

        return null_distribution_data


    ##################################
    ### SECTION: ADHERENCE SCORING ###
    ##################################


    def filter_by_cid(self, cid:str, digits:int = 3):

        cid = cid.strip().lower()[:digits]
        # rule_idx = list()
        actions_filtered = list()
        X_filtered = list()
        Y_filtered = list()

        for i, rule in enumerate(self.rules):

            rule_cid = rule.get('CID')

            if rule_cid == None:

                raise TypeError('Reference data for CID can not be a python None.')

            rule_cid = set([c.strip().lower()[:digits] for c in rule_cid])

            if cid in rule_cid:

                actions_filtered.append( self.actions[i].strip().lower() )

        for reference_data_X, reference_data_Y in zip(self.X, self.Y):

            if reference_data_Y.strip().lower() in actions_filtered:

                X_filtered.append(reference_data_X)
                Y_filtered.append(reference_data_Y)

        return (X_filtered, Y_filtered)


    def simple_scorer(self, x, cid:str, raw_output:bool = False, **kwargs) -> list:

        doc = self.nlp(x)
        cid = cid.strip().lower()
        output = list()

        tokens = set([
            self.nltk_stemmer(token.text.lower()) 
            for token in doc if token.pos_.strip().upper() in self.syntax_elements
        ])

        X_filtered, Y_filtered = self.filter_by_cid(cid = cid)

        protocol_set = set(Y_filtered)
        output = list()

        # aqui, assume que cada Y_i refere-se as frases dos red 
        # flags separadamente (i.e., uma a uma)
        for protocol_i in protocol_set:

            protocol_i_redflag_list = [
                redflag 
                for (redflag, protocol) in zip(X_filtered, Y_filtered)
                if protocol == protocol_i
            ]

            protocol_i_score_list = list()

            for j in range(len(protocol_i_redflag_list)):

                redflag_text_j = protocol_i_redflag_list[j].lower().split()
                score_j = 0

                for reference_token in redflag_text_j:

                    if reference_token in tokens:

                        score_j +=1

                protocol_i_score_list.append( (redflag_text_j, score_j) )

            if raw_output == True:

                output.append(
                    (protocol_i, protocol_i_redflag_list, protocol_i_score_list)
                )

            else:

                best_score = max(protocol_i_score_list, key = lambda x: x[1])

                best_score_normalized = round(best_score[1] / len(best_score[0]), 4)

                output.append(
                    ( protocol_i, best_score_normalized )
                )

        return output


    def levenshtein_scorer(self, x, cid:str, raw_output:bool = False, **kwargs) -> list:

        # tokens = x.strip().lower().split()
        doc = self.nlp(x)
        cid = cid.strip().lower()
        output = list()

        tokens = set([
            self.nltk_stemmer(token.text.lower()) 
            for token in doc if token.pos_.strip().upper() in self.syntax_elements
        ])

        X_filtered, Y_filtered = self.filter_by_cid(cid = cid)

        protocol_set = set(Y_filtered)
        output = list()

        # aqui, assume que cada Y_i refere-se as frases dos red 
        # flags separadamente (i.e., uma a uma)
        for protocol_i in protocol_set:

            protocol_i_redflag_list = [
                redflag 
                for (redflag, protocol) in zip(X_filtered, Y_filtered)
                if protocol == protocol_i
            ]

            protocol_i_score_list = list()

            for j in range(len(protocol_i_redflag_list)):

                redflag_text_j = protocol_i_redflag_list[j].lower().split()
                score_j = 0

                for reference_token in redflag_text_j:

                    similarity_scores = [Levenshtein.ratio(reference_token, token) for token in tokens]

                    if any([similarity_score > TOKEN_SIMILARITY_THRESHOLD for similarity_score in similarity_scores]):

                        score_j +=1

                protocol_i_score_list.append( (redflag_text_j, score_j) )

            if raw_output == True:

                output.append(
                    (protocol_i, protocol_i_redflag_list, protocol_i_score_list)
                )

            else:

                best_score = max(protocol_i_score_list, key = lambda x: x[1])

                best_score_normalized = round(best_score[1] / len(best_score[0]), 4)

                output.append(
                    ( protocol_i, best_score_normalized )
                )

        return output


    def jaccard_similarity(self, u:set, v:set) -> float:

        if not isinstance(u,set) or not isinstance(v,set):

            raise TypeError("'u' and 'v' must be python set")

        if len(u) == 0 or len(v) == 0:

            return np.nan

        else:

            return len(u.intersection(v)) / len(u.union(v))


    def jaccard_scorer(self, x, cid:str, raw_output:bool = False, **kwargs) -> list:

        doc = self.nlp(x)
        cid = cid.strip().lower()
        output = list()

        tokens_set = set([
            self.nltk_stemmer(token.text.lower()) 
            for token in doc if token.pos_.strip().upper() in self.syntax_elements
        ])

        X_filtered, Y_filtered = self.filter_by_cid(cid = cid)

        protocol_set = set(Y_filtered)
        output = list()

        # aqui, assume que cada Y_i refere-se as frases dos red 
        # flags separadamente (i.e., uma a uma)
        for protocol_i in protocol_set:

            protocol_i_redflag_list = [
                redflag 
                for (redflag, protocol) in zip(X_filtered, Y_filtered)
                if protocol == protocol_i
            ]

            protocol_i_score_list = list()

            for j in range(len(protocol_i_redflag_list)):

                redflag_text_j_set = set(protocol_i_redflag_list[j].lower().split())

                score_j = self.jaccard_similarity(redflag_text_j_set, tokens_set)

                score_j = (0 if np.isnan(score_j) else score_j)

                protocol_i_score_list.append( (redflag_text_j_set, score_j) )

            if raw_output == True:

                output.append(
                    (protocol_i, protocol_i_redflag_list, protocol_i_score_list)
                )

            else:

                # observacao: Levenshtein.setratio ja vem normalizado
                best_score = max(protocol_i_score_list, key = lambda x: x[1])

                output.append(
                    ( protocol_i, best_score[1] )
                )

        return output


    def set_bow(self):

        bow_object = self.get_bow()

        return bow_object['bow'], bow_object['tfidf_vectorizer']


    def imb_pipeline(self, X, y):

        model = Pipeline(
            [
                ('sampling', SMOTE(random_state=42, k_neighbors=1)), #ADASYN(random_state=42, n_neighbors=1)), # RandomUnderSampler(random_state=42)),
                ('model', MultinomialNB())
            ]
        )

        score={
            'AUC':'roc_auc',
            'RECALL':'recall',
            'PRECISION':'precision',
            'F1':'f1',
            'ACCURACY': 'accuracy'
        }

        params = [
            {'model__alpha': (0.001, 0.5, 1.0)}, 
            {'model__fit_prior': (True, False)}
        ]

        gcv = GridSearchCV(
            estimator           =   model,
            param_grid          =   params,
            cv                  =   3,
            scoring             =   score,
            n_jobs              =   None,
            refit               =   'ACCURACY', #'AUC', #'F1',
            return_train_score  =   True
        )

        gcv.fit(X, y)

        return gcv


    def get_models(self, bow):

        # TODO: implementar para multiplos modelos (reutilizar run_model_training da API do ChurnAnalyticsFramework)

        models = dict()

        protocols_set = set(self.Y)

        for protocol_i in protocols_set:

            y = [protocol_name == protocol_i for protocol_name in self.Y]

            model_i = self.imb_pipeline(bow, y)

            models[protocol_i] = model_i

        return models


    def get_bow(self):
        
        tfidf_vectorizer = TfidfVectorizer(
            strip_accents = 'ascii',
            lowercase = True,
            analyzer = 'word', #'char_wb',
            # stop_words = stopwords,
            ngram_range = (1, 3),

            # binary = False, #False,
            # norm = False,
            # use_idf = False,
            # sublinear_tf = False #True
        )
        tfidf_vectorizer.fit(self.X)

        bow = tfidf_vectorizer.transform(self.X)

        # tfidf_vectorizer.get_feature_names()

        return {
            'tfidf_vectorizer': tfidf_vectorizer,
            'bow': bow
        }


    def bow_scorer(self, x:str, cid:str, raw_output:bool = False):

        doc = self.nlp(x)
        cid = cid.strip().lower()
        output = list()

        x_tokenized = [
            ' '.join([
                self.nltk_stemmer(token.text.lower()) 
                for token in doc if token.pos_.strip().upper() in self.syntax_elements
            ])
        ]

        x_transformed = self.tfidf_vectorizer.transform(x_tokenized)
        _, Y_filtered = self.filter_by_cid(cid = cid)

        for protocol_i in set(Y_filtered):

            model_i = self.bow_models[protocol_i]

            if raw_output == True:

                output_i = model_i.predict_proba(x_transformed)

            elif raw_output == False:

                output_i = model_i.predict_proba(x_transformed)[0][1]

            else:

                raise TypeError("`raw_output` must be a python bool.")

            output.append(
                (protocol_i, output_i)
            )

        return output


    def bert_text_preparation(self, text, tokenizer):
        """Preparing the input for BERT
        
        Takes a string argument and performs
        pre-processing like adding special tokens,
        tokenization, tokens to ids, and tokens to
        segment ids. All tokens are mapped to seg-
        ment id = 1.
        
        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object
                to convert text into BERT-re-
                adable tokens and ids
            
        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids
        
        Reference:
            Prakash, A. 2021. 3 Types of Contextualized Word 
            Embeddings From BERT Using Transfer Learning.
            Medium Towards Data Science. Link: https://towardsdatascience.
            com/3-types-of-contextualized-word-embeddings-from-bert-
            using-transfer-learning-81fcefe3fe6d
        
        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors


    def get_bert_embeddings(self, tokens_tensor, segments_tensors, model):
        """Get embeddings from an embedding model
        
        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens]
                with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens]
                with segment ids for each token in text
            model (obj): Embedding model to generate embeddings
                from token and segment ids
        
        Returns:
            list: List of list of floats of size
                [n_tokens, n_embedding_dimensions]
                containing embeddings for each token
        
        Reference:
            Prakash, A. 2021. 3 Types of Contextualized Word 
            Embeddings From BERT Using Transfer Learning.
            Medium, Towards Data Science. Link: https://towardsdatascience.
            com/3-types-of-contextualized-word-embeddings-from-bert-
            using-transfer-learning-81fcefe3fe6d

        """
        
        # Gradient calculation id disabled
        # Model is in inference mode
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Removing the first hidden state
            # The first state is the input state
            # hidden_states = outputs[2][1:]
            hidden_states = outputs[0]

        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[:,1:-1,:] #[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings#, hidden_states


    def bert_scorer(self, x:str, cid:str, raw_output:bool = False):

        if raw_output == True:

            raise NotImplementedError('`raw_output` is not implemented for this method.')

        doc = self.nlp(x)
        cid = cid.strip().lower()
        output = list()

        redflags = list()
        labels = list()

        for i, rule in enumerate(self.rules):

            if cid in ' '.join(rule['CID']):

                redflags.extend(rule['REDFLAG'])
                labels.extend([self.actions[i]] * len(rule['REDFLAG']))

        if len(redflags) == 0 or len(labels) == 0:

            return output

        redflag_tokens = list()
        redflag_tokens_labels = list()

        for (redflag, label) in zip(redflags, labels):

            redflag = self.nlp(redflag)

            redflag_tokens_i = [
                token.text.strip().lower() 
                for token in redflag if token.pos_.strip().upper() in SYNTAX_ELEMENTS
            ]

            redflag_tokens_labels.extend( [label]*len(redflag_tokens_i) )

            redflag_tokens.extend(redflag_tokens_i)

        tokens = [
            token.text.lower()
            for token in doc if token.pos_.strip().upper() in SYNTAX_ELEMENTS
        ]

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        for token in tokens:

            tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(token, self.bert_tokenizer)
            token_embed =  self.get_bert_embeddings(tokens_tensor, segments_tensors, self.bert_model)

            token_embed = torch.tensor(token_embed).sum(axis=0)

            for (redflag_token, redflag_token_label) in zip(redflag_tokens, redflag_tokens_labels):

                tokenized_redflag_text, redflag_tokens_tensor, segments_redflag_tensors = self.bert_text_preparation(redflag_token, self.bert_tokenizer)
                redflag_token_embed =  self.get_bert_embeddings(redflag_tokens_tensor, segments_redflag_tensors, self.bert_model)

                redflag_token_embed = torch.tensor(redflag_token_embed).sum(axis=0)

                similarity_score = cos(redflag_token_embed, token_embed)

                output.append((redflag_token_label, redflag_token, token, similarity_score))


        output = sorted(output, key=lambda x: x[2], reverse=True)

        last_label = None
        output_clean = list()

        for i in range(len(output)):

            if (i < len(output) - 1) and (output[i][0] == output[i+1][0]):

                output_clean.append((output[i][0], output[i][2]))
                break
            
            else:

                if output[i][0] != last_label:

                    output_clean.append((output[i][0], output[i][2]))
                    last_label = output[i][0]

        return output_clean


    def sentence_transformer_scorer(self, x, cid:str, raw_output:bool = False):

        # TODO

        raise NotImplementedError('Sorry, not implemented for now.')


    def get_adherence(self, x:str, cid:str, method:int = 1, **kwargs) -> list:

        raw_output = kwargs.get('raw_output')
        raw_output = (False if raw_output is None else raw_output)

        if not isinstance(x, str):

            raise TypeError("`x` must be a python str.")

        if not isinstance(method, int):

            raise TypeError("`method` must be a python int.")

        if not isinstance(raw_output, bool):

            raise TypeError("`raw_output` must be a python bool.")


        x = x.strip().lower()


        if method == 0:

            return self.simple_scorer(x, cid=cid, raw_output=raw_output)

        elif method == 1:

            return self.levenshtein_scorer(x, cid=cid, raw_output=raw_output)

        elif method == 2:

            return self.jaccard_scorer(x, cid=cid, raw_output=raw_output)

        elif method == 3:

            return self.bow_scorer(x, cid=cid, raw_output=raw_output)
        
        elif method == 4:

            return self.bert_scorer(x, cid=cid, raw_output=raw_output)

        elif method == 5:

            return self.sentence_transformer_scorer(x, cid=cid, raw_output=raw_output)

        else:

            raise ValueError(
        "`method` must be either 0 (`simple_scorer` method) or 1 (`jaccard_scorer` method)."
        )


    #################################################
    ### SECTION: STATISTICAL SIGNIFICANCE TESTING ###
    #################################################


    def get_tokens_union_set(self):

        token_set = set()

        for redflag_tokens in self.X:

            redflag_tokens_set = set(redflag_tokens.split())
            token_set.update(redflag_tokens_set)

        return token_set


    def get_null_distribution(self, n_iterations:int = 1000):

        null_values = list()
        n_iterations = n_iterations
        counter = 0

        n_min = 1
        n_max = max([len(x.split()) for x in self.X]) + 1

        n_sample = random.sample(range(n_min, n_max+1), k=1)[0]

        union_set = self.get_tokens_union_set()

        while counter < n_iterations:

            random_redflag = random.sample(union_set, k=n_sample)
            random_redflag_str = ' '.join(random_redflag)

            random_adherence = self.get_adherence(random_redflag_str)
            null_values.append(random_adherence)

            counter += 1
        
        return null_values


    def p_value(self, observed_value:float, protocol:str):

        equal_or_greater = (self.null_distribution_data.query(f'protocol == "{protocol}"')['score'].values >= observed_value)

        if equal_or_greater is not None and len(equal_or_greater) > 0:

            return sum(equal_or_greater) / (len(equal_or_greater) + 0.001)
        
        else:

            return np.nan


############################


    def load_xlsx_2(self) -> None:

        protocol = list()
        # conditions = set()
        data = list()
        config = None

        xlsx_file = xl.readxl(fn=self.__filepath)
        sheet_names = xlsx_file.ws_names

        for sheet_name in sheet_names:

            if sheet_name.strip().lower() != 'config':

                # updating the set of decision table protocol
                # protocol.append(sheet_name.strip().upper())
                item = {'protocol': sheet_name.strip().upper()}

            sheet_data = xlsx_file.ws(ws=sheet_name)
            # item = dict()

            for i in range(1, len(sheet_data.row(row=1)) + 1):

                column_data = sheet_data.col(col=i)
                col_name = str(column_data[0]).strip().upper()
                col_contents = [
                    str(x).strip().lower()
                    for x in column_data[1:]
                    if not str(x).isspace() and not x == ''
                ]

                if (not col_name.isspace() and 
                    not col_name=='' and 
                    sheet_name.strip().lower() != 'config'):

                    # updating the set of decision table conditions
                    # conditions.update({col_name})

                    item[col_name] = self.set_datatype(col_name, col_contents)

                elif sheet_name.strip().lower() == 'config':

                    item[col_name] = col_contents

                else:

                    pass

            if sheet_name.strip().lower() != 'config':

                # updating the set of decision table redflag
                data.append(item)

            else:

                # updating the set of decision table config
                config = item


        # return protocol, conditions, redflag, config
        return data, config


    def update_bow_scorer(self):

        self.bow, self.tfidf_vectorizer = self.set_bow() 
        self.bow_models =  self.get_models(self.bow)


    def add_data(self, contents:list, protocol:str):

        if not isinstance(contents, list):

            raise TypeError('`contents` must be a python list (a list of strings).')

        if not isinstance(protocol, str):

            raise TypeError('`protocol` must be a python str.')

        X = list()
        Y = list()

        for content in contents:

            doc = self.nlp(content)
            x_i = set()

            for token in doc:

                if token.pos_.strip().upper() in self.syntax_elements:

                    # usando lematização
                    # x_i.update({token.lemma_.lower()})

                    # usando stematização
                    x_i.update({ self.nltk_stemmer(token.text.strip().lower()) })

            X.append(' '.join(x_i))
            Y.append(protocol)

        self.X.extend(X)
        self.Y.extend(Y)

        self.update_bow_scorer()
