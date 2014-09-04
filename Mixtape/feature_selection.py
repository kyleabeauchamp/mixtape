import sklearn
import numpy as np
import mixtape.tica
import mixtape.subset_featurizer


def clone_and_swap(featurizer):
    """Clone a featurizer and randomly swap one of its features.
    
    Parameters
    ----------
    featurizer : SubsetUnionFeaturizer
        The featurizer to clone and swap.
    
    Returns
    -------
    featurizer : SubsetUnionFeaturizer
        A new featurizer with one of its features swapped with a randomly selected feature.
    """
    featurizer = sklearn.clone(featurizer)
    new_feature = np.random.choice(featurizer.n_featurizers)
    new_value = np.random.choice(featurizer.n_max_i[new_feature])
    
    current_feature_weights = featurizer.n_features_i / (1.0 * featurizer.n_features_i.sum())
    
    featurizer_to_replace = np.random.choice(featurizer.n_featurizers, p=current_feature_weights)
    
    featurizer_index_to_replace = np.random.choice(featurizer.n_features_i[featurizer_to_replace])
    
    print("%s %d %d %d %d" % (" " * 80, featurizer_to_replace, featurizer_index_to_replace, new_feature, new_value))
    
    if featurizer_to_replace == new_feature and new_value == featurizer_index_to_replace:
        print('no swap')
        return featurizer
    
    if new_value in featurizer.transformer_list[new_feature][1].subset:
        print('no swap (already have feature)')
        return featurizer        
    
    value_to_remove = featurizer.transformer_list[featurizer_to_replace][1].subset[featurizer_index_to_replace]
    set0 = set(featurizer.transformer_list[featurizer_to_replace][1].subset)
    set0.remove(value_to_remove)
    featurizer.transformer_list[featurizer_to_replace][1].subset = np.array(list(set0))
    
    set1 = set(featurizer.transformer_list[new_feature][1].subset)
    set1.add(new_value)
    featurizer.transformer_list[new_feature][1].subset = np.array(list(set1))

    return featurizer


class Optimizer(object):
    """Optimize TICA objective function by swapping active features one-by-one."""
    def __init__(self, featurizer, model):
        
        self.featurizer = featurizer
        self.model = model
    
    def optimize(self, n_iter, trajectories):
        """Optimize TICA objective function by random swapping.
        
        Parameters
        ----------
        n_iter : int
            Number of iterations to attempt
        trajectories : list of md.Trajectory
            Trajectories to use.
        """
        
        features = self.featurizer.transform(trajectories)
        self.model = sklearn.clone(self.model)
        self.model.fit(features)
        self.current_score = self.model.score(features)
        
        for i in range(n_iter):
            new_featurizer = clone_and_swap(self.featurizer)
            
            features = new_featurizer.transform(trajectories)
            new_model = sklearn.clone(self.model)
            new_model.fit(features)
            new_score = new_model.score(features)

            if new_score > self.current_score:
                accept = True
            else:
                accept = False

            print("%d * %.4f %.4f" % (accept, self.current_score, new_score))
            #print("%d * %.4f %.4f ** %.4f %.4f %.4f **** %.4f %.4f %.4f" % (
            #accept, self.current_score, new_score, self.model.eigenvalues_[0], self.model.eigenvalues_[1], self.model.eigenvalues_[2], new_model.eigenvalues_[0], new_model.eigenvalues_[1], new_model.eigenvalues_[2]))
            
            if accept:
                self.model = new_model
                self.featurizer = new_featurizer
                self.current_score = new_score                
        
        return self.featurizer
