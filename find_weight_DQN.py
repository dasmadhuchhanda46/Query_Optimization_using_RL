import random
import numpy as np
from collections import deque, Counter
from sklearn.neural_network import MLPRegressor
from typing import List
from lucene import initVM
from org.apache.lucene.search import IndexSearcher, TermQuery, BooleanQuery, BooleanClause, BoostQuery
from org.apache.lucene.index import Term, DirectoryReader
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.document import Document
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from java.io import File, StringReader
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute

class QueryWeightOptimizer:
    def __init__(self, stopword_path, index_dir: str, query_id=601, term_count=10, num_episodes=100, max_steps=50, gamma=0.9, batch_size=32):
        self.stopword_path = stopword_path
        self.field_name = "TEXT"
        self.reader = self.init_reader(index_dir)
        self.searcher = self.init_searcher()
        self.query_id = query_id
        self.terms = self.expanded_terms(query_id)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_terms = len(self.terms)
        self.weights = np.full(self.num_terms, 0.5)
        
        # DQN components using sklearn MLPRegressor
        self.q_network = MLPRegressor(hidden_layer_sizes=(64,), learning_rate_init=0.01, max_iter=1, warm_start=True)
        self.q_network.fit([np.zeros(self.num_terms)], [np.zeros(self.num_terms)])  # Dummy fit for initialization
        self.memory = deque(maxlen=1000)
        self.weight_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # Discrete weight values

    def init_reader(self, index_dir: str):
        """Initialize the Lucene index reader."""
        directory = FSDirectory.open(File(index_dir).toPath())  # Open the directory containing the index
        return DirectoryReader.open(directory)  # Create and return a DirectoryReader instance    
    
    def init_searcher(self):
        """Initialize the Lucene index searcher."""
        return IndexSearcher(self.reader)  # Create and return an IndexSearcher instance

    def expanded_terms(self, query_id: int, number_terms = 10) -> List[str]:
        relevence_file = "/home/madhu/lucene_code/trec_data/lucene_data/trec678rb/qrels/robust_601-700.qrel"
        analyzer = EnglishAnalyzer()
        term_counter = Counter()
        with open(relevence_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                current_query_id, doc_id, relevance =  int(parts[0].strip()), parts[2], int(parts[3])
                if query_id == current_query_id:
                    if relevance > 0:
                        self.query_rel_doc_map.append(doc_id)
                        doc = None
                        for i in range(self.reader.maxDoc()):
                            document = self.reader.document(i)
                            if document.get("DOCNO") == doc_id: 
                                doc = document
                                break
                        if doc:
                            content = doc.get(self.field_name)
                            token_stream = analyzer.tokenStream("", StringReader(content))
                            try:
                                token_stream.reset()
                                while token_stream.incrementToken():
                                    term = token_stream.getAttribute(CharTermAttribute.class_).toString()
                                    term_counter[term] += 1
                            finally:
                                token_stream.close()
        _terms = [word for word, _ in  term_counter.most_common(number_terms)]
        print(f"extended query terms: \n{_terms}")
        return _terms

    def create_query(self, weights):
        """
        Create a weighted Boolean query based on the current weights.

        :param weights: List of weights for the query terms.
        :return: Lucene BooleanQuery object.
        """
        query_builder = BooleanQuery.Builder()
        for i, term in enumerate(self.terms):
            term_query = TermQuery(Term(self.field_name, term))
            # boosted_query = BooleanClause(term_query, BooleanClause.Occur.SHOULD)
            # boosted_query.boost = weights[i]
            # query_builder.add(boosted_query)
            boosted_query = BoostQuery(term_query, float(weights[i]))
            query_builder.add(boosted_query, BooleanClause.Occur.SHOULD)
        return query_builder.build()


    def read_relevance_docs(self, qrel_path: str):
        """Read judged relevant and non-relevant documents from the QREL file."""
        with open(qrel_path, "r") as file:  # Open the QREL file
            for line in file:  # Process each line in the file
                tokens = line.split()  # Split the line into tokens
                query_id, doc_id, relevance = tokens[0], tokens[2], tokens[3]  # Extract query ID, document ID, and relevance
                if relevance == "0":  # If the document is non-relevant
                    self.query_non_rel_doc_map[query_id].append(doc_id)  # Add to the non-relevant map
                else:  # If the document is relevant
                    self.query_rel_doc_map[query_id].append(doc_id)  # Add to the relevant map


    def evaluate_query(self, query):
        """
        Compute the Mean Average Precision (MAP) for a query.

        Parameters:
        - query: The original query.
        - rocchio_vector: List of RocchioStat objects for expanded terms.
        - qrel_path: Path to the QREL file.

        Returns:
        - MAP value for the given query.
        """
        # expanded_query = self.expanded_query(query, rocchio_vector, alpha=1.0)  # Generate the expanded query
        top_docs = self.searcher.search(query, 1000)  # Retrieve the top 1000 results

        # Extract the document IDs of the retrieved results
        retrieved_docs = [self.searcher.doc(score_doc.doc).get("DOCNO") for score_doc in top_docs.scoreDocs]
        relevant_docs = set(self.query_rel_doc_map)  # Get the relevant documents for the query

        if not relevant_docs:  # If there are no relevant documents, return 0 MAP
            return 0.0

        # Calculate precision at each relevant document's rank
        precision_at_k = []
        relevant_count = 0
        for rank, docid in enumerate(retrieved_docs, start=1):
            if docid in relevant_docs:  # Check if the document is relevant
                relevant_count += 1
                precision_at_k.append(relevant_count / rank)  # Compute precision at this rank

        # Compute and return the Mean Average Precision
        return sum(precision_at_k) / len(relevant_docs) # if relevant_docs else 0.0

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return np.random.choice(self.weight_values, size=self.num_terms)
        else:
            q_values = self.q_network.predict([state])[0]
            return np.array([self.weight_values[np.argmax(q_values[i])] for i in range(self.num_terms)])
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_dqn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        target_q_values = rewards + self.gamma * np.max(self.q_network.predict(next_states), axis=1) * (1 - dones)
        self.q_network.fit(states, target_q_values)
    
    def train(self):
        for episode in range(self.num_episodes):
            print(f"Starting episode {episode + 1}/{self.num_episodes}")
            self.weights = np.full(self.num_terms, 0.5)
            for step in range(self.max_steps):
                state = self.weights.copy()
                action = self.select_action(state)
                self.weights = action  # Directly assign the discrete action values
                reward = self.evaluate_query(self.create_query(self.weights))
                next_state = self.weights.copy()
                done = step == self.max_steps - 1
                self.store_experience(state, action, reward, next_state, done)
                self.train_dqn()
            print(f"Episode {episode + 1} completed. Final Weights: {self.weights}")
        print("Training completed!")

initVM()
indx = "/home/madhu/lucene_code/index_trec"
optimizer = QueryWeightOptimizer("stopword_path", indx)
optimizer.train()
print("Optimized Weights:", optimizer.weights)
