import random
import copy

class Qrels():
    def __init__(self, irrelevant=True,
                 use_inverse_dictionary=True,
                 getter_expression="qi -> q, dr, dn",
                 filepath=None):
        
        self.irrelevant = irrelevant

        self.use_inverse_dictionary = use_inverse_dictionary

        self.query_document_relevant = {}
        self.query_document_irrelevant = {}

        self.document_query_relevant = {}
        self.document_query_irrelevant = {}
    
        self._query_index = []
        self._document_index = []

        self.get_mod = getter_expression
        self.set_getter(self.get_mod)
        if(filepath is not None):
            self.set_ressource(filepath)
    @property
    def use_inverse_dictionary(self) -> bool:
        return  self._use_inverse_dictionary
    
    @use_inverse_dictionary.setter
    def use_inverse_dictionary(self, value: bool) -> None:
        if("_use_inverse_dictionary" not in self.__dict__):
            self._use_inverse_dictionary = value
        if value == self._use_inverse_dictionary:
            return
        else:
            if(value):
                self.document_query_relevant, self.document_query_irrelevant = \
                    self._process_document_query_dictionary()
            else:
                self.document_query_relevant = None
                self.document_query_irrelevant = None

        self._use_inverse_dictionary = value
    
    def get_dictionary(self):
       return { k: {d: 1 for d in v} for k, v in self.query_document_relevant.items()}

    def _process_document_query_dictionary(self):

        relevant =\
            Qrels._process_inverse_dictionary(self.query_document_relevant)
        irrelevant = {}
        if(self.irrelevant):
            irrelevant =\
                Qrels._process_inverse_dictionary(self.query_document_irrelevant)

        return relevant, irrelevant

    @staticmethod
    def _process_inverse_dictionary(source: dict) -> dict:
        target = {}
        for k, vs in source.items():
            for v in vs:
                if v not in target:
                    target[v] = set()
                target[v].add(k)
        return target

    @staticmethod
    def _merge_(source: dict, new_content:dict) -> None:
        for k, v in new_content.items():
            if(k not in source):
                source[k] = set()
            source[k].update(v)

    @staticmethod
    def _merge(source: dict, new_content:dict) -> dict:
        new_source = copy.deepcopy(source)
        Qrels._merge_(new_source, new_content)
        return new_source

    @staticmethod
    def _load_qrels_file(filepath, store_irrelevant=True):
        relevant, irrelevant = {}, {}

        with open(filepath) as qrels_file:
            while True:
                line = qrels_file.readline()
                if not line:
                    break
                qrel = line.split()
                query_id = qrel[0]
                document_id = qrel[2]
                relevance = int(qrel[3])

                if relevance == 1:
                    if(query_id not in relevant):
                        relevant[query_id] = set()
                    relevant[query_id].add(document_id)
                
                elif(relevance == 0 and store_irrelevant):
                    if(query_id not in irrelevant):
                        irrelevant[query_id] = set()
                    irrelevant[query_id].add(document_id)

        return relevant, irrelevant
    
    def _update_query_index(self, queries):
        for k in queries:
            self._query_index.append(k)


    def add_ressource(self, filepath_or_qrels) -> None:
        if(isinstance(filepath_or_qrels, str)):
            relevant, irrelevant =\
                Qrels._load_qrels_file(filepath_or_qrels, self.irrelevant)
        else:
            relevant = filepath_or_qrels.query_document_relevant
            irrelevant = filepath_or_qrels.query_document_irrelevant

        Qrels._merge_(self.query_document_relevant, relevant)
        Qrels._merge_(self.query_document_irrelevant, irrelevant)
        self._update_query_index(relevant)
        self._update_query_index(irrelevant)

        if(self.use_inverse_dictionary):
            relevant_inversed =\
                Qrels._process_inverse_dictionary(relevant)
            Qrels._merge_(self.document_query_relevant, relevant_inversed)
            if(self.irrelevant):
                irrelevant_inversed =\
                    Qrels._process_inverse_dictionary(irrelevant)
                Qrels._merge_(self.document_query_irrelevant, irrelevant_inversed)

    @staticmethod
    def merge_qrels(*args):
        qrels = Qrels()
        for arg in args:
            qrels.add_ressource(arg)
        return qrels

    def set_ressource(self, filepath: str) -> None:
        self.query_document_relevant = {}
        self.query_document_irrelevant = {}

        self.document_query_relevant = {}
        self.document_query_irrelevant = {}

        self.add_ressource(filepath)

    def get_qdr(self, index):
        if(index in self.query_document_relevant):
            return self.query_document_relevant[index]
        else:
            return set()

    def get_qdi(self, index):
        if(index in self.query_document_irrelevant):
            return self.query_document_irrelevant[index]
        else:
            return set()

    def set_getter(self, expression) -> None:
        self.input_getter, self.output_getters =\
            self._create_getter(expression)

    def __contains__(self, query_id):
        return (query_id in self.query_document_relevant or 
                query_id in self.query_document_irrelevant)

    def _create_getter(self, expression):
        expression = expression.replace(' ', '')
        splitted = expression.split('->')
        if len(splitted) > 2 :
            raise Exception("Only one input is allowed")
        input_token = splitted[0]
        output_tokens = splitted[1].split(',')
        if "q" in input_token:
            # input
            input_getter = lambda x: x
            
            if "i" in input_token:
                input_getter = lambda x: self._query_index[x]
            
            output_getters = []
            # output
            for output_token in output_tokens:
                output_getter = lambda x: None
                if("d" in output_token):
                    output_getter =\
                        lambda x: self.get_qdr(x).union(self.get_qdi(x))
                    if("r" in output_token):
                        output_getter = self.get_qdr
                    if("n" in output_token):
                        output_getter = self.get_qdi
                if "q" in output_token:
                    # input
                    output_getter = lambda x: x
                    
                    if "i" in output_token:
                        output_getter = lambda x: x
                output_getters.append(output_getter)
            
            return input_getter, output_getters

    def __getitem__(self, index):
       return tuple([ output_getter(self.input_getter(index)) for output_getter in self.output_getters])

    def __len__(self):
        return len(self._query_index)


# class TSVQueryRankingFile():
#     def __init__(self, filepath, structure=["query_id", "document_id", "score"]):
#         for 
