# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import glob
import numpy as np
import mdtraj as md
import msmbuilder as msmb

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class Directory(object):
    def __init__(self, pdb_filename, trj_path, extension="h5", stride=1):
        """A container for a directory of Trajectory objects.

        Parameters
        ----------
        pdb_filename : string
            Path to a single PDB object that will provide topology information
        trj_path : string
            Path to a directory containing trajectory objects
        stride : int, default=1, optional
            Only read every stride-th frame.        
        """
        self.trj0 = md.load(pdb_filename)
        self.trj_path = trj_path
        self.filenames = sorted(glob.glob(trj_path + "/*.%s" % extension), key=msmb.utils.keynat)
        self.trajectories = None
        self.stride = stride
        self.index = 0
        
    def cache(self):
        """Load all trajectories into memory."""
        self.trajectories = [md.load(filename, stride=self.stride) for filename in self.filenames]

    def featurize_all(self, featurizer):
        """Featurize all trajectory files.
        
        Parameters
        ----------
        featurizer : Featurizer
            The featurizer to be invoked on each trajectory trajectory as
            it is loaded
        Returns
        -------
        data : np.ndarray, shape=(total_length_of_all_trajectories, n_features)
        indices : np.ndarray, shape=(total_length_of_all_trajectories)
        fns : np.ndarray shape=(total_length_of_all_trajectories)
            These three arrays all share the same indexing, such that data[i] is
            the featurized version of indices[i]-th frame in the MD trajectory
            with filename fns[i].
        """
        data = []
        indices = []
        fns = []
        
        stride = 1  # Hardcoded right now.

        for k, trj in enumerate(self):

            count = 0

            x = featurizer.featurize(trj)
            n_frames = len(x)

            data.append(x)
            indices.append(count + (stride*np.arange(n_frames)))
            fns.extend([self.filenames[k]] * n_frames)
            count += (stride*n_frames)
        
        if len(data) == 0:
            raise ValueError("None!")

        return np.concatenate(data), np.concatenate(indices), np.array(fns)

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()
    
    def __next__(self):
        try:
            result = self[self.index]
        except IndexError:
            self.index = 0  # Automatically reset index to allow repeated iterations
            raise StopIteration
        self.index += 1
        return result

    def __getitem__(self, index):
        if self.trajectories is None:
            return md.load(self.filenames[index], top=self.trj0)
        else:
            return self.trajectories[index]
