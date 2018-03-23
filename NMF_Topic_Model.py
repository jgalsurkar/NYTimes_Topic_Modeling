import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class NMF_Topic_Model(object):
    """Nonnegative Matrix Factorization based Topic Modeling
    
    Parameters
    ----------        
    term_doc : array-like, shape = [num_vocab, num_documents]
        wordcount document matrix

    num_vocab : Integer
        Vocabulary size
        
    num_documents : Integer
        Number of documents
    
    rank : Integer
        Number of topics to learn
    """
    
    def __init__(self, term_doc, num_vocab, num_documents, rank):
        self.term_doc = term_doc
        self.W = np.random.rand(num_vocab, rank)
        self.H = np.random.rand(rank, num_documents)
        self.objectives = []
        
    def update_data_matrix(self):
        """ Update term data matrix
        
        Returns
        ----------
        X_div : array-like
            term data matrix
        """
        WH = np.dot(self.W, self.H)
        X_div = np.divide(self.term_doc, WH)
        x_nans = np.isnan(X_div)
        X_div[x_nans] = 0

        return X_div
    
    def update_H(self):
        """ Update H matrix (rank X M)
        """
        X_div = self.update_data_matrix()
        normalized_rows = normalize(self.W.T, norm = 'l1', axis = 1)
        matrix_mul = np.matmul(normalized_rows, X_div)
        self.H = np.multiply(self.H, matrix_mul)
        
    def update_W(self):
        """ Update W matrix (N X rank)
        """
        X_div = self.update_data_matrix()
        normalized_cols = normalize(self.H.T, norm = 'l1', axis = 0)
        matrix_mul = np.matmul(X_div, normalized_cols)
        self.W = np.multiply(self.W, matrix_mul)
        
    def calculate_objective(self):
        """ Calculate the current value of the objective function
        
        Returns
        ----------
        objective_val : Float
            objective function value based on current WH
        """
        WH = np.dot(self.W, self.H)
        interaction = np.multiply(self.term_doc, np.log(WH))
        nans = np.isnan(interaction)
        interaction[nans] = 0
        objective_val = -np.subtract(interaction, WH).sum()
        
        return objective_val
        
    def nmf(self, num_iterations):
        """ Perform coordinate ascent and track objective values
        """
        for i in range(num_iterations):
            self.update_H() 
            self.update_W()
            self.objectives.append(self.calculate_objective())
            
    def get_topics(self, vocabulary, num_words):
        """ Get the topics and top words associated with them
        
        Parameters
        ----------
        vocabulary : array-like
            List of words, index corresponds to term_doc
        
        num_words : Integer
            Number of words to return with each topic
        
        Returns
        ----------
        topic_top_words : array-like
            Top words and corresponding weight for each topic
        """
        w = pd.DataFrame(normalize(self.W, norm='l1', axis=0))
        topic_top_words = []
        i = 0
        for topic in w:
            top_words = pd.DataFrame(w[topic]).sort_values(by = topic, ascending = False)[:num_words]
            top_words = top_words.assign(Topic = vocabulary.iloc[top_words.index])
            top_words.rename(columns = {i:'Weight'}, inplace = True)
            top_words.reset_index(inplace = True, drop = True)
            top_words.index += 1
            topic_top_words.append(top_words[['Topic', 'Weight']])
            i+=1
            
        return topic_top_words