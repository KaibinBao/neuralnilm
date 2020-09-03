from __future__ import print_function, division
import numpy as np
import pandas as pd
from datetime import timedelta
from neuralnilm.data.source import Sequence
from neuralnilm.data.activationssource import ActivationsSource
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class ActivationOnlySource(ActivationsSource):
    def __init__(self, activations, target_appliance, seq_length,
                 sample_period,
                 distractor_inclusion_prob=0.25,
                 target_inclusion_prob=0.5,
                 uniform_prob_of_selecting_each_building=True,
                 allow_nonactive_target=False,
                 allow_nonactive_distractors=True,
                 include_incomplete_target_in_output=True,
                 rng_seed=None):
        self.activations = activations
        self.target_appliance = target_appliance
        self.seq_length = seq_length
        self.sample_period = sample_period
        self.distractor_inclusion_prob = distractor_inclusion_prob
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (
            uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = True # don't let ActivationSource filter out activations
        self.allow_nonactive_target = allow_nonactive_target
        self.allow_nonactive_distractors = allow_nonactive_distractors
        self.include_incomplete_target_in_output = (
            include_incomplete_target_in_output)
        super(ActivationOnlySource, self).__init__(rng_seed=rng_seed)


    def _get_sequence(self, fold='train', enable_all_appliances=False):
        seq = Sequence(self.seq_length)
        all_appliances = {}

        building_i = 0
        # Target appliance
        if self.rng.binomial(n=1, p=self.target_inclusion_prob):
            building_name, building_i = self._select_building_with_index(fold, self.target_appliance)
            activations = (
                self.activations[fold][self.target_appliance][building_name])
            activation_i = self._select_activation(activations)
            activation = activations[activation_i]
            positioned_activation, is_complete, _ = self._position_incomplete_activation(
                activation, is_target_appliance=True)
            if enable_all_appliances:
                all_appliances[self.target_appliance] = positioned_activation
            if is_complete or self.include_incomplete_target_in_output:
                seq.target += positioned_activation
            building_i += 1

        seq.input = seq.target[:, np.newaxis]
        seq.target = seq.target[:, np.newaxis]
        #seq.weights = np.float32([building_i])[:, np.newaxis]
        assert len(seq.target) == self.seq_length        
        if enable_all_appliances:
            seq.all_appliances = pd.DataFrame(all_appliances)
        return seq


    def _select_building_with_index(self, fold, appliance):
        """
        Parameters
        ----------
        fold : str
        appliance : str

        Returns
        -------
        building_name : str
        """
        if fold not in DATA_FOLD_NAMES:
            raise ValueError("`fold` must be one of '{}' not '{}'."
                             .format(DATA_FOLD_NAMES, fold))

        activations_per_building = self.activations[fold][appliance]

        # select `p` for np.random.choice
        if self.uniform_prob_of_selecting_each_building:
            p = None  # uniform distribution over all buildings
        else:
            num_activations_per_building = np.array([
                len(activations) for activations in
                activations_per_building.values()])
            p = (num_activations_per_building /
                 num_activations_per_building.sum())

        building_names = list(activations_per_building.keys())
        num_buildings = len(building_names)
        building_i = self.rng.choice(num_buildings, p=p)
        building_name = building_names[building_i]
        return building_name, building_i


    def _position_incomplete_activation(self, activation, is_target_appliance):
        """
        Parameters
        ----------
        activation : pd.Series
        is_target_appliance : bool

        Returns
        -------
        positioned_activation : np.array
        is_complete : whether the activation is completely contained within the sequence
        seq_start_time : datetime when the sequence starts
        """
        if is_target_appliance:
            allow_nonactive = self.allow_nonactive_target
        else:
            allow_nonactive = self.allow_nonactive_distractors

        # Select a start index
        if allow_nonactive:
            earliest_start_i = -len(activation)
            latest_start_i = self.seq_length
        else:
            # original:
            #if len(activation) > self.seq_length:
            #    raise RuntimeError("Activation too long to fit into sequence"
            #                       " and incomplete activations not allowed.")
            #earliest_start_i = 0
            #latest_start_i = self.seq_length - len(activation)
            # modified:
            activation_length = len(activation)
            earliest_start_i = min(-activation_length+self.seq_length, 0)
            latest_start_i = max(0, self.seq_length-activation_length)

        start_i = self.rng.randint(low=earliest_start_i, high=latest_start_i)

        positioned_activation = np.zeros(self.seq_length)

        # Clip or pad head of sequence
        len_activation = len(activation.values)
        if start_i < 0:
            remaining = len_activation + start_i
            # Clip or pad tail to produce a sequence which is seq_length long
            if remaining <= self.seq_length:
                positioned_activation[0:remaining] = activation.values[-start_i:remaining-start_i]
            else:
                positioned_activation = activation.values[-start_i:self.seq_length-start_i]
            is_complete = False
        else:
            # Clip or pad tail to produce a sequence which is seq_length long
            activation_end = len_activation + start_i
            if activation_end <= self.seq_length:
                positioned_activation[start_i:activation_end] = activation.values
                is_complete = True
            else:
                activation_end = min(activation_end, self.seq_length)
                positioned_activation[start_i:activation_end] = activation.values[:activation_end-start_i]
                is_complete = False

        if len(activation) > self.seq_length:
            assert(is_complete == False)
        else:
            space_after_activation = self.seq_length - len(activation)
            assert(is_complete == (0 <= start_i <= space_after_activation))

        seq_start_time = activation.index[0] - timedelta(
            seconds=start_i * self.sample_period)
        return positioned_activation, is_complete, seq_start_time