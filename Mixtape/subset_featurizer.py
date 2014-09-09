import itertools
import numpy as np
import mixtape.featurizer, mixtape.tica
import mdtraj as md

ATOM_NAMES = ["N", "CA", "CB", "C", "O", "H"]

def get_atompair_indices(reference_traj, keep_atoms=ATOM_NAMES, exclude_atoms=None, reject_bonded=True):
    """Get a list of acceptable atom pairs.

    Parameters
    ----------
    reference_traj : mdtraj.Trajectory
        Trajectory to grab atom pairs from
    keep_atoms : np.ndarray, dtype=string, optional
        Select only these atom names
    exclude_atoms : np.ndarray, dtype=string, optional
        Exclude these atom names
    reject_bonded : bool, default=True
        If True, exclude bonded atompairs.  
        
    Returns
    -------
    atom_indices : np.ndarray, dtype=int
        The atom indices that pass your criteria
    pair_indices : np.ndarray, dtype=int, shape=(N, 2)
        Pairs of atom indices that pass your criteria.

    Notes
    -----
    This function has been optimized for speed.  A naive implementation
    can be slow (~minutes) for large proteins.
    """
    top, bonds = reference_traj.top.to_dataframe()
    
    if keep_atoms is not None:
        atom_indices = top[top.name.isin(keep_atoms) == True].index.values
    
    if exclude_atoms is not None:
        atom_indices = top[top.name.isin(exclude_atoms) == False].index.values

    pair_indices = np.array(list(itertools.combinations(atom_indices, 2)))

    if reject_bonded:
        a_list = bonds.min(1)
        b_list = bonds.max(1)

        n = atom_indices.max() + 1

        bond_hashes = a_list + b_list * n
        pair_hashes = pair_indices[:, 0] + pair_indices[:,1] * n

        not_bonds = ~np.in1d(pair_hashes, bond_hashes)

        pair_indices = np.array([(a, b) for k, (a, b) in enumerate(pair_indices) if not_bonds[k]])

    
    return atom_indices, pair_indices


def _lookup_pairs_subset(all_pair_indices, subset_pair_indices, n_choose=None):
    """Convert pairs of atom indices into a list of indices

    Parameters
    ----------
    all_pair_indices : np.ndarray, dtype='int', shape=(N, 2)
        All allowed pairs of atom indices
    subset_pair_indices : np.ndarray, dtype=int, shape=(n, 2)
        A select subset of the atom pairs
    n_choose : int, default=None
        if not None, return at most this many indices

    Returns
    -------
    subset : np.ndarray, dtype=int, shape=(n)
        A numpy array with the integer indices that map subset_pair_indices
        onto all_pair_indices.  That is, subset[k] indices the value of 
        all_pair_indices that matches subset_pair_indices[k] 
        
    
    Notes
    -----
    This function is mostly useful when you have two lists of atom_pair
    indices and you want to find "indices" mapping the smaller to the larger
    list.  This could occur when you are looking at different atom_pair
    featurizers.
    
    """


    n = all_pair_indices.max()
    
    all_keys = all_pair_indices[:, 0] + n * all_pair_indices[:, 1]
    optimal_keys = subset_pair_indices[:, 0] + n * subset_pair_indices[:, 1]
    subset = np.where(np.in1d(all_keys, optimal_keys))[0]

    if n_choose is not None:
        subset[0:min(len(subset), n_choose)] = subset[0:min(len(subset), n_choose)]

    return subset

def guess_featurizers(trj0, n_choose, rmsd_trajectory=None):
    """Get a SubsetUnionFeaturizer that includes "best-practices" features.
    
    Parameters
    ----------
    trj0 : md.Trajectory
        Reference Trajectory, used for looking up dihedrals.
    n_choose : int
        Number of features to activate.  
    rmsd_trajectory : MD.Trajectory, optional, default=None
        If given, also use the RMSD to the frames in this trajectory as
        features.  For example, this trajectory could be a single crystal
        structure or a trajectory of cluster centers.
    
    Returns
    -------
    featurizer : SubsetUnionFeaturizer
        Returns a featurizer that includes best-practices features that
        can be enabled or disabled via subsets.
    
    """
    trj0 = trj0[0]  # Slice the first frame of the reference trajectory to avoid huge memory consumption.
    
    atom_indices, pair_indices = get_atompair_indices(trj0, keep_atoms=None, exclude_atoms=np.array([]))
    
    atom_featurizer1m = SubsetAtomPairs(pair_indices, trj0, exponent=-1.0)
    atom_featurizer2m = SubsetAtomPairs(pair_indices, trj0, exponent=-2.0)
    atom_featurizer1p = SubsetAtomPairs(pair_indices, trj0, exponent= 1.0)
    
    prod_featurizer = SubsetProductFeaturizer(SubsetAtomPairs(pair_indices, trj0, exponent=-1.0), SubsetAtomPairs(pair_indices, trj0, exponent=-1.0))
    
    cosphi = SubsetCosPhiFeaturizer(trj0)
    sinphi = SubsetSinPhiFeaturizer(trj0)
    cospsi = SubsetCosPsiFeaturizer(trj0)
    sinpsi = SubsetSinPsiFeaturizer(trj0)

    if rmsd_trajectory is None:
        featurizer = SubsetFeatureUnion([("pairs1m", atom_featurizer1m), ("pairs2m", atom_featurizer2m), ("pairs1p", atom_featurizer1p), ("prod1m", prod_featurizer), ("cosphi", cosphi), ("sinphi", sinphi), ("cospsi", cospsi), ("sinpsi", sinpsi)])
    else:
        rmsd_featurizer = SubsetRMSDFeaturizer(rmsd_trajectory)
        featurizer = SubsetFeatureUnion([("pairs1m", atom_featurizer1m), ("pairs2m", atom_featurizer2m), ("pairs1p", atom_featurizer1p), ("prod1m", prod_featurizer), ("cosphi", cosphi), ("sinphi", sinphi), ("cospsi", cospsi), ("sinpsi", sinpsi), ("rmsd", rmsd_featurizer)])
    
    subsets = [[] for i in range(featurizer.n_featurizers)]
    subsets[0] = np.random.randint(0, atom_featurizer1m.n_max - 1, n_choose)
    featurizer.subsets = subsets    

    return featurizer


class BaseSubsetFeaturizer(mixtape.featurizer.Featurizer):
    """Base class for featurizers that have a subset of active features.
    n_features refers to the number of active features.  n_max refers to the 
    number of possible features.

    Parameters
    ----------
    reference_traj : mdtraj.Trajectory
        Reference Trajectory for checking consistency
    subset : np.ndarray, default=None, dtype=int
        The values in subset specify which of all possible features 
    
    Notes
    -----
    
    As an example, suppose we have an instance that has `n_max` = 5.  This
    means that the possible features are subsets of [0, 1, 2, 3, 4].  One possible
    subset is then [0, 1, 3].  The allowed values of subset (e.g. `n_max`)
    will be determined by the subclass--e.g. for example, `n_max` might be
    the number of phi backbone angles.
    """

    def __init__(self, reference_traj, subset=None):
        self.reference_traj = reference_traj
        if subset is not None:
            self.subset = subset
        else:
            self.subset = np.zeros(0, 'int')

    @property
    def n_features(self):
        return len(self.subset)


class SubsetAtomPairs(BaseSubsetFeaturizer):
    """Subset featurizer based on atom pair distances.

    Parameters
    ----------
    possible_pair_indices : np.ndarray, dtype=int, shape=(n_max, 2)
        These are the possible atom indices to use for calculating interatomic
        distances.  
    reference_traj : mdtraj.Trajectory
        Reference Trajectory for checking consistency
    subset : np.ndarray, default=None, dtype=int
        The values in subset specify which of all possible features are
        to be enabled.  Specifically, atom pair distances are calculated
        for the pairs `possible_pair_indices[subset]`
    periodic : bool, optional, default=False
        if True, use periodic boundary condition wrapping
    exponent : float, optional, default=1.0
        Use the distances to this power as the output feature.
    
    See Also
    --------
    
    See `get_atompair_indices` for how one might generate acceptable atom pair
    indices.
    
    """    
    def __init__(self, possible_pair_indices, reference_traj, subset=None, periodic=False, exponent=1.0):
        super(SubsetAtomPairs, self).__init__(reference_traj, subset=subset)
        self.possible_pair_indices = possible_pair_indices
        self.periodic = periodic
        self.exponent = exponent
        if subset is None:
            self.subset = np.zeros(0, 'int')
        else:
            self.subset = subset
            

    @property
    def n_max(self):
        return len(self.possible_pair_indices)

    def partial_transform(self, traj):
        if self.n_features > 0:
            features = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic) ** self.exponent
        else:
            features = np.zeros((traj.n_frames, 0))
        return features

    @property
    def pair_indices(self):
        return self.possible_pair_indices[self.subset]


class SubsetTrigFeaturizer(BaseSubsetFeaturizer):
    """Base class for featurizer based on dihedral sine or cosine.
    
    Notes
    -----
    
    Subsets must be a subset of 0, ..., n_max - 1, where n_max is determined
    by the number of respective phi / psi dihedrals in your protein, as
    calcualted by mdtraj.compute_phi and mdtraj.compute_psi
    
    """    
    
    def partial_transform(self, traj):
        if self.n_features > 0:
            dih = md.geometry.dihedral.compute_dihedrals(traj, self.which_atom_ind[self.subset])
            features = self.trig_function(dih)
        else:
            features = np.zeros((traj.n_frames, 0))
        return features

    @property
    def n_max(self):
        return len(self.which_atom_ind)

class CosMixin(object):
    def trig_function(self, dihedrals):
        return np.cos(dihedrals)

class SinMixin(object):
    def trig_function(self, dihedrals):
        return np.sin(dihedrals)

class PhiMixin(object):
    @property
    def which_atom_ind(self):
        atom_indices, dih = md.geometry.dihedral.compute_phi(self.reference_traj)
        return atom_indices

class PsiMixin(object):
    @property
    def which_atom_ind(self):
        atom_indices, dih = md.geometry.dihedral.compute_psi(self.reference_traj)
        return atom_indices
    

class SubsetCosPhiFeaturizer(SubsetTrigFeaturizer, CosMixin, PhiMixin):
    pass


class SubsetCosPsiFeaturizer(SubsetTrigFeaturizer, CosMixin, PhiMixin):
    pass
    

class SubsetSinPhiFeaturizer(SubsetTrigFeaturizer, SinMixin, PsiMixin):
    pass
    

class SubsetSinPsiFeaturizer(SubsetTrigFeaturizer, SinMixin, PsiMixin):
    pass


class SubsetProductFeaturizer(BaseSubsetFeaturizer):
    """Subset Featurizer that gives elementwise proudcts of two featurizers.
    
    Parameters
    ----------
    featurizer0 : SubsetFeaturizer
        First featurizer
    featurizer1 : SubsetFeaturizer
        Second featurizer
    """
    def __init__(self, featurizer0, featurizer1):
        self.featurizer0 = featurizer0
        self.featurizer1 = featurizer1

    def partial_transform(self, traj):
        return np.array(self.featurizer0.partial_transform(traj) * self.featurizer1.partial_transform(traj))

    @property
    def n_max(self):
        return self.featurizer0.n_max * self.featurizer1.n_max

    @property
    def cumprod(self):
        if self._cumprod:
            return self._cumprod
        else:
            self._cumprod = np.cumprod([f.n_max for f in self.featurizers]) / self.featurizers[0].n_max

    @property
    def subset(self):
        return self.featurizer0.subset * self.featurizer0.n_max + self.featurizer1.subset

    @subset.setter
    def subset(self, value):
        value = np.array(value)
        self.featurizer0.subset = value / self.featurizer0.n_max
        self.featurizer1.subset = value % self.featurizer1.n_max 



class SubsetRMSDFeaturizer(BaseSubsetFeaturizer):
    """Featurizer based on RMSD to a series of reference frames.

    Parameters
    ----------
    trj0 : mdtraj.Trajectory
        Reference trajectory.  trj0.n_frames gives the number of features
        in this Featurizer.
    atom_indices : np.ndarray, default=None
        Which atom indices to use during RMSD calculation.  If None, MDTraj
        should default to all atoms.

    """

    def __init__(self, trj0, atom_indices=None, subset=None):
        self.trj0 = trj0
        self.atom_indices = atom_indices
        if subset is not None:
            self.subset = subset
        else:
            self.subset = np.zeros(0, 'int')


    @property
    def n_max(self):
        return self.trj0.n_frames

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space by calculating
        the RMSD to each frame in a reference trajectory.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        X = np.zeros((traj.n_frames, self.n_features))

        for frame in range(self.n_features):
            X[:, frame] = md.rmsd(traj, self.trj0[self.subset], atom_indices=self.atom_indices, frame=frame)
        return X


class SubsetFeatureUnion(mixtape.featurizer.TrajFeatureUnion):
    """Mixtape version of sklearn.pipeline.FeatureUnion with feature subset selection.
    
    Notes
    -----
    Works on lists of trajectories.
    Has a hacky convenience method to set all subsets at once.
    """

    @property
    def subsets(self):
        return [featurizer.subset for (_, featurizer) in self.transformer_list]

    @subsets.setter
    def subsets(self, value):
        assert len(value) == len(self.transformer_list), "wrong len"
        for k, (_, featurizer) in enumerate(self.transformer_list):
            featurizer.subset = value[k]


    @property
    def n_max_i(self):
        return np.array([featurizer.n_max for (_, featurizer) in self.transformer_list])

    @property
    def n_features_i(self):
        return np.array([featurizer.n_features for (_, featurizer) in self.transformer_list])

    @property
    def n_featurizers(self):
        return len(self.transformer_list)

    @property
    def n_max(self):
        return np.sum([featurizer.n_max for (_, featurizer) in self.transformer_list])

    @property
    def n_features(self):
        return sum([featurizer.n_features for (_, featurizer) in self.transformer_list])


class DummyCV(object):
    """A cross-validation object that returns identical training and test sets."""
    def __init__(self, n):
        self.n = n

    def __iter__(self):
            yield np.arange(self.n), np.arange(self.n)

    def __len__(self):
        return self.n
