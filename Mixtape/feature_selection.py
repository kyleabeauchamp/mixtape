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


class TICAScoreMixin(object):
    """Provides compare() and summarize() functionality for TICAOptimizer.
    Uses variational eigenvalue comparison to rank models.
    """
    def compare(self):
        self.accept = is_better(self.model.eigenvalues_, self.old_model.eigenvalues_)
        self.summarize()
        return self.accept

    def summarize(self):
        print("%d %.5f %.4f %.4f **** %.5f %.4f %.4f" % (
        self.accept, self.old_model.eigenvalues_[0], self.old_model.eigenvalues_[1], self.old_model.eigenvalues_[2], self.model.eigenvalues_[0], self.model.eigenvalues_[1], self.model.eigenvalues_[2]))


class TICAOptimizer(TICAScoreMixin):
    """Optimize TICA objective function by swapping active features one-by-one."""
    def __init__(self, featurizer, lag_time=1):
        
        self.featurizer = featurizer
        self.lag_time = lag_time


    def _build(self, featurizer, trajectories):
        """Featurize all active subsets and build a tICA model."""
        tica = mixtape.tica.tICA(lag_time=self.lag_time)
        features = featurizer.transform(trajectories)
        unused_output = tica.fit(features)
        
        return tica, 1.0 # tica.score(features)

    
    def optimize(self, n_iter, trajectories):
        """Optimize TICA objective function by random swapping.
        
        Parameters
        ----------
        n_iter : int
            Number of iterations to attempt
        trajectories : list of md.Trajectory
            Trajectories to use.
        """
        
        self.model, self.obj = self._build(self.featurizer, trajectories)
        self.old_model = self.model
        self.old_obj = self.obj

        
        for i in range(n_iter):
            new_featurizer = clone_and_swap(self.featurizer)
            self.model, self.obj = self._build(new_featurizer, trajectories)
            if not self.compare():
                self.model = self.old_model
            else:
                self.featurizer = new_featurizer
            self.old_model = self.model
            self.old_obj = self.obj


def is_better(lam, lam0):
    """Compares lists of ordered eigenvalues for slowness."""
    try:
        first_gain = np.where(lam > lam0)[0][0]
        first_loss = np.where(lam < lam0)[0][0]
        return first_gain < first_loss
    except:
        return False
