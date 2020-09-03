from __future__ import print_function, division
from copy import copy
import gc
from datetime import timedelta
import numpy as np
import pandas as pd
import nilmtk
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.timeframe import TimeFrame
from nilmtk.utils import timedelta64_to_secs
from neuralnilm.data.source import Sequence
from neuralnilm.data.source import Source
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class RealAggregateActivityData():
    """
    Attributes
    ----------
    data : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <building_name>: pd.Series of raw data
        }}
    data_good_sections : dict
        Same structure as `mains`.
    sections_with_no_target : dict
        Same structure as `mains`.
    all_gaps : dict of pd.DataFrames
        Each key is a fold name.
        Each DF has columns:
        building, gap, duration, p (short for 'probability')
        p is used by _get_sequence_without_target().
    """
    def __init__(self, target_appliance, nilmtk_activations,
                 nilmtk_data, sample_period,
                 activation_max_stale_pct=0.81,
                 mains_max_stale_pct=0.68):
        self.target_appliance = target_appliance
        self.sample_period = sample_period

        self.data = nilmtk_data['data']
        self.data_good_sections = nilmtk_data['good_sections']

        self.activation_max_stale_pct = activation_max_stale_pct
        self.mains_max_stale_pct = mains_max_stale_pct

        self._classify_activation_quality(nilmtk_activations)
        self._delete_phony_sections()
        self._remove_timeframes_with_no_data()

    def _classify_activation_quality(self, nilmtk_activations):
        def get_stale_seconds(act):
            actdiff = act.resample("{:d}S".format(self.sample_period)).mean().ffill().diff()
            return (actdiff == 0.0).sum() * self.sample_period

        def activation_filter(tf, building_data):
            start_time = tf.start
            end_time = tf.end
            df = building_data[start_time:end_time]
            if df.empty:
                return False
            else:
                act_stale_seconds = get_stale_seconds(df['target'])
                act_duration = (end_time - start_time).total_seconds()
                act_stale_pct = act_stale_seconds / act_duration
                mains_stale_seconds = get_stale_seconds(df['mains'])
                mains_stale_pct = get_stale_seconds(df['mains']) / act_duration
                if (act_stale_pct < self.activation_max_stale_pct) & (mains_stale_pct < self.mains_max_stale_pct):
                    return True
                else:
                    return False

        good_timeframes = {}
        bad_timeframes = {}
        all_timeframes = {}
        for fold, buildings_per_appliances in nilmtk_activations.items():
            good_timeframes[fold] = {}
            bad_timeframes[fold] = {}
            all_timeframes[fold] = {}
            for appliance, activations_per_building in buildings_per_appliances.items():
                good_timeframes[fold][appliance] = {}
                bad_timeframes[fold][appliance] = {}
                all_timeframes[fold][appliance] = {}
                for building, activations in activations_per_building.items():
                    building_data = self.data[fold][building]
                    good_timeframes_per_building = TimeFrameGroup()
                    bad_timeframes_per_building = TimeFrameGroup()
                    all_timeframes_per_building = TimeFrameGroup()
                    for i, activation in enumerate(activations):
                        tf = TimeFrame(
                            start=activation.index[0],
                            end=activation.index[-1] + pd.Timedelta(seconds=self.sample_period))
                        all_timeframes_per_building.append(tf)
                        if activation_filter(tf, building_data):
                            good_timeframes_per_building.append(tf)
                        else:
                            bad_timeframes_per_building.append(tf)
                    good_timeframes[fold][appliance][building] = good_timeframes_per_building
                    bad_timeframes[fold][appliance][building] = bad_timeframes_per_building
                    all_timeframes[fold][appliance][building] = all_timeframes_per_building
        #
        self.clean_active_timeframes = good_timeframes
        self.all_active_timeframes = all_timeframes
        self.phony_active_timeframes = bad_timeframes


    def _delete_phony_sections(self):
        filtered_data = {}
        for fold, data_per_building in self.data.items():
            for building, data in data_per_building.items():
                if building not in self.phony_active_timeframes[fold][self.target_appliance]:
                    continue
                activations = (
                    self.phony_active_timeframes[fold][self.target_appliance][building])
                data_between_phony_activations = TimeFrameGroup()
                prev_end = data.index[0]
                for activation in activations:
                    activation_start = activation.start
                    if prev_end < activation_start:
                        gap = TimeFrame(prev_end, activation_start)
                        data_between_phony_activations.append(gap)
                    prev_end = activation.end
                data_end = data.index[-1] + pd.Timedelta(seconds=self.sample_period)
                if prev_end < data_end:
                    gap = TimeFrame(prev_end, data_end)
                    data_between_phony_activations.append(gap)
                dfs = []
                for section in data_between_phony_activations:
                    dfs.append(section.slice(data))
                data = pd.concat(dfs)
                filtered_data.setdefault(fold, {})[building] = (
                    data)
                logger.info("Found {} good sections for {} {}."
                            .format(len(data_between_phony_activations), fold, building))

        self.data = filtered_data

    def _foreach_fold_and_building(self, data_dict, func):
        result = {}
        for fold, stuff_per_building in data_dict.items():
            result[fold] = {}
            for building, stuff in stuff_per_building.items():
                result[fold][building] = func(stuff)
        return result

    def _get_good_sections(self, df):
        index = df.dropna().sort_index().index
        df_time_end = df.index[-1] + pd.Timedelta(seconds=self.sample_period)
        del df

        if len(index) < 2:
            return []

        timedeltas_sec = timedelta64_to_secs(np.diff(index.values))
        timedeltas_check = timedeltas_sec <= self.sample_period

        # Memory management
        del timedeltas_sec
        gc.collect()

        timedeltas_check = np.concatenate(
            [[False],
             timedeltas_check])
        transitions = np.diff(timedeltas_check.astype(np.int))

        # Memory management
        last_timedeltas_check = timedeltas_check[-1]
        del timedeltas_check
        gc.collect()

        good_sect_starts = list(index[:-1][transitions ==  1])
        good_sect_ends   = list(index[:-1][transitions == -1])

        # Memory management
        last_index = index[-1]
        del index
        gc.collect()
        
        # Work out if this chunk ends with an open ended good section
        if len(good_sect_ends) == 0:
            ends_with_open_ended_good_section = (
                len(good_sect_starts) > 0 or 
                previous_chunk_ended_with_open_ended_good_section)
        elif len(good_sect_starts) > 0:
            # We have good_sect_ends and good_sect_starts
            ends_with_open_ended_good_section = (
                good_sect_ends[-1] < good_sect_starts[-1])
        else:
            # We have good_sect_ends but no good_sect_starts
            ends_with_open_ended_good_section = False

        if ends_with_open_ended_good_section:
            good_sect_ends += [df_time_end]

        assert len(good_sect_starts) == len(good_sect_ends)

        sections = [TimeFrame(start, end)
                    for start, end in zip(good_sect_starts, good_sect_ends)
                    if not (start == end and start is not None)]

        # Memory management
        del good_sect_starts
        del good_sect_ends
        gc.collect()

        return sections

    def _has_sufficient_samples(self, data, start, end, threshold=0.8):
        if len(data) < 2:
            return False
        num_expected_samples = (
            (end - start).total_seconds() / self.sample_period)
        hit_rate = len(data) / num_expected_samples
        return (hit_rate >= threshold)

    def _remove_timeframes_with_no_data(self):
        # First remove any activations where there is no mains data at all
        for fold, activations_for_appliance in self.clean_active_timeframes.items():
            activations_for_buildings = activations_for_appliance[
                self.target_appliance]
            buildings_to_remove = []
            for building in activations_for_buildings:
                data_for_fold = self.data[fold]
                if (building not in data_for_fold and
                        building not in buildings_to_remove):
                    buildings_to_remove.append(building)
            for building in buildings_to_remove:
                self.clean_active_timeframes[fold][self.target_appliance].pop(building)

        # Now check for places where mains has insufficient samples,
        # for example because the mains series has a break in it.
        for fold, activations_for_appliance in self.clean_active_timeframes.items():
            activations_for_buildings = activations_for_appliance[
                self.target_appliance]
            buildings_to_remove = []
            for building, activations in activations_for_buildings.items():
                data = self.data[fold][building]
                activations_to_remove = []
                for i, activation in enumerate(activations):
                    activation_duration = activation.timedelta
                    start = activation.start - activation_duration
                    end = activation.end + activation_duration
                    data_for_activ = data[start:end]
                    if (start < data.index[0] or
                            end > data.index[-1] or not
                            self._has_sufficient_samples(
                                data_for_activ, start, end)):
                        activations_to_remove.append(i)
                if activations_to_remove:
                    logger.info(
                        "Removing {} activations from fold '{}' building '{}'"
                        " because there was not enough mains data for"
                        " these activations. This leaves {} activations."
                        .format(
                            len(activations_to_remove), fold, building,
                            len(activations) - len(activations_to_remove)))
                activations_to_remove.reverse()
                for i in activations_to_remove:
                    activations.pop(i)
                if len(activations) == 0:
                    buildings_to_remove.append(building)
                else:
                    self.clean_active_timeframes[fold][self.target_appliance][building] = (
                        activations)
            for building in buildings_to_remove:
                self.clean_active_timeframes[fold][self.target_appliance].pop(building)


class BalancedActivityRealAggregateSource(Source):
    """
    Attributes
    ----------
    data : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <building_name>: pd.Series of raw data
        }}
    data_good_sections : dict
        Same structure as `mains`.
    sections_with_no_target : dict
        Same structure as `mains`.
    all_gaps : dict of pd.DataFrames
        Each key is a fold name.
        Each DF has columns:
        building, gap, duration, p (short for 'probability')
        p is used by _get_sequence_without_target().
    """
    def __init__(self, activity_data, seq_length,
                 target_inclusion_prob=0.5,
                 uniform_prob_of_selecting_each_building=True,
                 allow_incomplete_target=True,
                 vampire_power_per_building=[],
                 rng_seed=None):
        super(BalancedActivityRealAggregateSource, self).__init__(rng_seed=rng_seed)

        self.activations = activity_data.clean_active_timeframes
        self.all_activations = activity_data.all_active_timeframes
        self.target_appliance = activity_data.target_appliance
        self.seq_length = seq_length
        self.sample_period = activity_data.sample_period
        self.data = activity_data.data
        self.data_good_sections = activity_data.data_good_sections
        self.vampire_power_per_building = vampire_power_per_building
        self._find_sections_with_no_target()
        self._compute_gap_probabilities()

        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (
            uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = allow_incomplete_target

    def _find_sections_with_no_target(self):
        """Finds the intersections of the mains good sections with the gaps
        between target appliance activations.
        """
        self.sections_with_no_target = {}
        seq_length_secs = self.seq_length * self.sample_period
        for fold, sects_per_building in self.data_good_sections.items():
            for building, good_sections in sects_per_building.items():
                if building not in self.all_activations[fold][self.target_appliance]:
                    continue
                activations = (
                    self.all_activations[fold][self.target_appliance][building])
                data = self.data[fold][building]
                data_good_sections = good_sections
                gaps_between_activations = TimeFrameGroup()
                prev_end = data.index[0]
                for activation in activations:
                    activation_start = activation.start
                    if prev_end < activation_start:
                        gap = TimeFrame(prev_end, activation_start)
                        gaps_between_activations.append(gap)
                    prev_end = activation.end
                data_end = data.index[-1]
                if prev_end < data_end:
                    gap = TimeFrame(prev_end, data_end)
                gaps_between_activations.append(gap)
                intersection = (
                    gaps_between_activations.intersection(data_good_sections))
                intersection = intersection.remove_shorter_than(
                    seq_length_secs)
                self.sections_with_no_target.setdefault(fold, {})[building] = (
                    intersection)
                logger.info("Found {} sections without target for {} {}."
                            .format(len(intersection), fold, building))

    def _compute_gap_probabilities(self):
        # Choose a building and a gap
        self.all_gaps = {}
        for fold in list(self.sections_with_no_target.keys()):
            all_gaps_for_fold = []
            for building, gaps in self.sections_with_no_target[fold].items():
                gaps_for_building = [
                    (building, gap, gap.timedelta.total_seconds())
                    for gap in gaps]
                all_gaps_for_fold.extend(gaps_for_building)
            gaps_df = pd.DataFrame(
                all_gaps_for_fold, columns=['building', 'gap', 'duration'])
            gaps_df['p'] = gaps_df['duration'] / gaps_df['duration'].sum()
            self.all_gaps[fold] = gaps_df

    def _get_sequence_without_target(self, fold):
        # Choose a building and a gap
        all_gaps_for_fold = self.all_gaps[fold]
        n = len(all_gaps_for_fold)
        assert(n != 0)
        gap_i = self.rng.choice(n, p=all_gaps_for_fold['p'])
        row = all_gaps_for_fold.iloc[gap_i]
        building, gap = row['building'], row['gap']

        # Choose a start point in the gap
        latest_start_time = gap.end - timedelta(
            seconds=self.seq_length * self.sample_period)
        max_offset_seconds = (latest_start_time - gap.start).total_seconds()
        if max_offset_seconds <= 0:
            offset = 0
        else:
            offset = self.rng.randint(max_offset_seconds)
        start_time = gap.start + timedelta(seconds=offset)
        data = self.data[fold][building]
        start_i, _ = data.index.slice_locs(start=start_time)

        #end_time = start_time + timedelta(
        #    seconds=(self.seq_length + 1) * self.sample_period)
        #data = self.data[fold][building][start_time:end_time]
        seq = Sequence(self.seq_length)
        #seq.input = data['mains'].values[:self.seq_length]
        #seq.target = data['target'].values[:self.seq_length]
        seq.input = data['mains'].values[start_i:start_i+self.seq_length]
        seq.target = data['target'].values[start_i:start_i+self.seq_length]
        if building in self.vampire_power_per_building:
            seq.input = np.clip(
                seq.input-self.vampire_power_per_building[building],
                0, None)
        if True: # add metadata ?
            seq.metadata['start'] = start_time
            seq.metadata['fold'] = fold
            seq.metadata['building'] = building
        return seq

    def get_sequence(self, fold='train', enable_all_appliances=False):
        while True:
            yield self._get_sequence(
                fold=fold, enable_all_appliances=enable_all_appliances)

    def _get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for BalancedActivityRealAggregateSource!")

        if(self.all_gaps[fold].empty): # check if there are any sequences without target
            _seq_getter_func = self._get_sequence_which_includes_target
        else:
            if self.rng.binomial(n=1, p=self.target_inclusion_prob):
                _seq_getter_func = self._get_sequence_which_includes_target
            else:
                _seq_getter_func = self._get_sequence_without_target

        MAX_RETRIES = 50
        for retry_i in range(MAX_RETRIES):
            seq = _seq_getter_func(fold=fold)
            if seq is None:
                continue
            if len(seq.input) != self.seq_length:
                continue
            if len(seq.target) != self.seq_length:
                continue
            break
        else:
            raise RuntimeError("No valid sequences found after {} retries!"
                               .format(MAX_RETRIES))

        seq.input = seq.input[:, np.newaxis]
        seq.target = seq.target[:, np.newaxis]
        assert len(seq.input) == self.seq_length
        assert len(seq.target) == self.seq_length
        return seq

    def _select_building(self, fold, appliance):
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
        return building_name

    def _select_activation(self, activations):
        num_activations = len(activations)
        if num_activations == 0:
            raise RuntimeError("No appliance activations.")
        activation_i = self.rng.randint(low=0, high=num_activations)
        return activation_i

    def _select_start_time(self, activation, is_target_appliance):
        """
        Parameters
        ----------
        activation : pd.Series
        is_target_appliance : bool

        Returns
        -------
        pd.Series
        """
        if is_target_appliance:
            allow_incomplete = self.allow_incomplete_target
        else:
            allow_incomplete = self.allow_incomplete_distractors

        activation_samples = int(activation.timedelta.total_seconds() // self.sample_period)

        # Select a start index
        if allow_incomplete:
            earliest_start_i = -activation_samples
            latest_start_i = self.seq_length
        else:
            if activation_samples > self.seq_length:
                #raise RuntimeError("Activation too long to fit into sequence"
                #                   " and incomplete activations not allowed.")
                #print("WARNING: Activation too long to fit into sequence.")
                earliest_start_i = -(activation_samples - self.seq_length)
                latest_start_i = 0
            else:
                earliest_start_i = 0
                latest_start_i = self.seq_length - activation_samples

        if earliest_start_i < latest_start_i:
            start_i = self.rng.randint(low=earliest_start_i, high=latest_start_i)
        else:
            #print(earliest_start_i, latest_start_i, activation_samples, self.seq_length)
            start_i = 0

        seq_start_time = activation.start - timedelta(
            seconds=start_i * self.sample_period)
        return seq_start_time

    def _get_sequence_which_includes_target(self, fold):
        seq = Sequence(self.seq_length)
        building_name = self._select_building(fold, self.target_appliance)
        activations = (
            self.activations[fold][self.target_appliance][building_name])
        activation_i = self._select_activation(activations)
        activation = activations[activation_i]
        activation_start_time = (
            self._select_start_time(
                activation, is_target_appliance=True))

        data_start = activation_start_time
        #data_end   = activation_start_time + timedelta(
        #    seconds=self.sample_period * (self.seq_length-1))

        # Get data
        data_for_building = self.data[fold][building_name]
        # load some additional data to make sure we have enough samples
        #data_end_extended = data_end + timedelta(
        #    seconds=self.sample_period * 2)
        #data = data_for_building[data_start:data_end_extended]
        start_i, _ = data_for_building.index.slice_locs(start=data_start)
        seq.input  = data_for_building['mains'].values[start_i:start_i+self.seq_length]
        seq.target = data_for_building['target'].values[start_i:start_i+self.seq_length]
        if building_name in self.vampire_power_per_building:
            seq.input = np.clip(
                seq.input-self.vampire_power_per_building[building_name],
                0, None)
        return seq

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return [
            'activations', 'rng', 'mains', 'mains_good_sections',
            'sections_with_no_target', 'all_gaps']


class BalancedActivityAugmentedAggregateSource(BalancedActivityRealAggregateSource):
    def __init__(self, activity_data, seq_length,
                 target_inclusion_prob=0.5,
                 uniform_prob_of_selecting_each_building=True,
                 allow_incomplete_target=True,
                 vampire_power_per_building=[],
                 rng_seed=None):
        super(BalancedActivityAugmentedAggregateSource, self).__init__(
            activity_data=activity_data,
            seq_length=seq_length,
            target_inclusion_prob=target_inclusion_prob,
            uniform_prob_of_selecting_each_building=(
                uniform_prob_of_selecting_each_building),
            allow_incomplete_target=(
                allow_incomplete_target),
            vampire_power_per_building=(
                vampire_power_per_building),
            rng_seed=rng_seed)

    def _get_sequence_with_target_in_another_building(self, fold):
        # Get data without target from another building
        seq = self._get_sequence_without_target(fold)
        building_name = self._select_building(fold, self.target_appliance)
        activations = (
            self.activations[fold][self.target_appliance][building_name])
        activation_i = self._select_activation(activations)
        activation = activations[activation_i]
        activation_start_time = (
            self._select_start_time(
                activation, is_target_appliance=True))

        # Get data
        data_start = activation_start_time
        #data_end   = activation_start_time + timedelta(
        #    seconds=self.sample_period * (self.seq_length-1))
        data_for_building = self.data[fold][building_name]
        # load some additional data to make sure we have enough samples
        #data_end_extended = data_end + timedelta(
        #    seconds=self.sample_period * 2)

        start_i, _ = data_for_building.index.slice_locs(start=data_start)
        target_data = data_for_building['target'].values[start_i:start_i+self.seq_length]

        if len(seq.input) != len(target_data): # input size missmatch, retry!
            return None
        seq.input = np.copy(seq.input)
        seq.input  -= seq.target # subtract any residual device activity (e.g., standby)
        seq.input  += target_data
        if building_name in self.vampire_power_per_building:
            seq.input -= self.vampire_power_per_building[building_name]
        seq.input = seq.input.clip(0, None)
        seq.target = target_data
        seq.metadata['activation_start'] = data_start
        seq.metadata['activation_building'] = building_name

        return seq

    def _get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for BalancedRealAggregateSource!")

        if(self.all_gaps[fold].empty): # check if there are any sequences without target
            _seq_getter_func = self._get_sequence_with_target_in_another_building
        else:
            if self.rng.binomial(n=1, p=self.target_inclusion_prob):
                _seq_getter_func = self._get_sequence_with_target_in_another_building
            else:
                _seq_getter_func = self._get_sequence_without_target

        MAX_RETRIES = 50
        for retry_i in range(MAX_RETRIES):
            seq = _seq_getter_func(fold=fold)
            if seq is None:
                continue
            if len(seq.input) != self.seq_length:
                continue
            if len(seq.target) != self.seq_length:
                continue
            break
        else:
            raise RuntimeError("No valid sequences found after {} retries!"
                               .format(MAX_RETRIES))

        seq.input = seq.input[:, np.newaxis]
        seq.target = seq.target[:, np.newaxis]
        assert len(seq.input) == self.seq_length
        assert len(seq.target) == self.seq_length
        return seq


class RandomizedSequentialSource(Source):
    def __init__(self, activity_data, seq_length,
                 stride=None,
                 vampire_power_per_building=[],
                 rng_seed=None):

        super(RandomizedSequentialSource, self).__init__(rng_seed=rng_seed)

        self.target_appliance = activity_data.target_appliance
        self.seq_length = seq_length
        self.sample_period = activity_data.sample_period
        self.data = activity_data.data
        self.data_good_sections = activity_data.data_good_sections
        self.stride = self.seq_length if stride is None else stride
        self.vampire_power_per_building = vampire_power_per_building
        self._reset()

        if (stride > seq_length):
            raise ValueError("`stride` should not be greater than `seq_length` ")
        
        self._compute_num_sequences_per_building()

    def _reset(self):
        self._num_seqs = pd.Series()

    def _compute_num_sequences_per_building(self):
        index = []
        all_num_seqs = []
        for fold, buildings in self.data.items():
            for building_name, df in buildings.items():
                remainder = len(df) - self.seq_length
                num_seqs = np.ceil(remainder / self.stride) + 1
                num_seqs = max(0 if df.empty else 1, int(num_seqs))
                if num_seqs > 0:
                    index.append((fold, building_name))
                    all_num_seqs.append(num_seqs)
        multi_index = pd.MultiIndex.from_tuples(
            index, names=["fold", "building_name"])
        self._num_seqs = pd.Series(all_num_seqs, multi_index)

    def total_sequences_for_fold(self, fold='train'):
        total_seq_for_fold = self._num_seqs[fold].sum()
        return total_seq_for_fold

    def get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for RandomizedSequentialSource!")

        # select building
        #building_divisions = self._num_seqs[fold].cumsum()
        total_seq_for_fold = self._num_seqs[fold].sum()

        building_base_seq_i = self._num_seqs[fold].cumsum()
        building_base_seq_i.values[1:] = building_base_seq_i.values[:-1]
        building_base_seq_i.values[0] = 0
        building_base_seq_i = pd.Series(
            building_base_seq_i.index.values,
            index=building_base_seq_i.values)

        for seq_i in self.rng.permutation(total_seq_for_fold):
            building_row_i = building_base_seq_i.index.get_loc(seq_i, method="ffill")
            building_name = building_base_seq_i.values[building_row_i]
            base_seq_i = building_base_seq_i.index[building_row_i]

            seq_i_for_building = seq_i - base_seq_i
            start_i = seq_i_for_building * self.stride
            end_i = start_i + self.seq_length
            dataframe = self.data[fold][building_name]
            columns = dataframe.columns
            data_for_seq = dataframe.values[start_i:end_i]

            def get_data(col):
                col_i = columns.get_loc(col)
                data = data_for_seq[:,col_i]
                len_data = len(data)
                zero_padded_data = np.zeros((self.seq_length, 1))
                zero_padded_data[:len_data,0] = data
                return zero_padded_data

            seq = Sequence(self.seq_length)
            seq.input = get_data('mains')
            seq.target = get_data('target')
            if building_name in self.vampire_power_per_building:
                seq.input = np.clip(
                    seq.input-self.vampire_power_per_building[building_name],
                    0, None)
            assert len(seq.input) == self.seq_length
            assert len(seq.target) == self.seq_length

            # Set mask
            seq.weights = np.ones((self.seq_length, 1), dtype=np.float32)
            n_zeros_to_pad = self.seq_length - len(data_for_seq)
            if n_zeros_to_pad > 0:
                seq.weights[-n_zeros_to_pad:, 0] = 0

            # Set metadata
            seq.metadata = {
                'seq_i': seq_i,
                'building_name': building_name,
                'total_num_sequences': total_seq_for_fold,
                # this takes a lot of time:
                'start_date': dataframe.index[start_i],
                'end_date': dataframe.index[start_i+len(data_for_seq)-1]
            }

            yield seq

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['data', '_num_seqs', 'rng']


class BalancedBuildingRealAggregateSource(Source):
    def __init__(self, activity_data, seq_length,
                 stride=None,
                 vampire_power_per_building=[],
                 rng_seed=None):

        super(BalancedBuildingRealAggregateSource, self).__init__(rng_seed=rng_seed)

        self.target_appliance = activity_data.target_appliance
        self.seq_length = seq_length
        self.sample_period = activity_data.sample_period
        self.data = activity_data.data
        self.data_good_sections = activity_data.data_good_sections
        self.stride = self.seq_length if stride is None else stride
        self.vampire_power_per_building = vampire_power_per_building
        self._reset()

        if (stride > seq_length):
            raise ValueError("`stride` should not be greater than `seq_length` ")
        
        self._compute_num_sequences_per_building()

    def _reset(self):
        self._num_seqs = pd.Series()

    def _compute_num_sequences_per_building(self):
        index = []
        all_num_seqs = []
        for fold, buildings in self.data.items():
            for building_name, df in buildings.items():
                remainder = len(df) - self.seq_length
                num_seqs = np.ceil(remainder / self.stride) + 1
                num_seqs = max(0 if df.empty else 1, int(num_seqs))
                if num_seqs > 0:
                    index.append((fold, building_name))
                    all_num_seqs.append(num_seqs)
        multi_index = pd.MultiIndex.from_tuples(
            index, names=["fold", "building_name"])
        self._num_seqs = pd.Series(all_num_seqs, multi_index)

    def total_sequences_for_fold(self, fold='train'):
        total_seq_for_fold = self._num_seqs[fold].sum()
        return total_seq_for_fold


    def get_sequence(self, fold='train', enable_all_appliances=False):
        while True:
            yield self._get_sequence(
                fold=fold, enable_all_appliances=enable_all_appliances)

    def _get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for BalancedActivityRealAggregateSource!")

        building_names = list(self.data[fold].keys())
        num_buildings = len(building_names)
        building_i = self.rng.randint(low=0, high=num_buildings)
        building_name = building_names[building_i]

        seq_i_for_building = self.rng.randint(
            low=0, high=self._num_seqs[(fold, building_name)])
        start_i = seq_i_for_building * self.stride
        end_i = start_i + self.seq_length

        dataframe = self.data[fold][building_name]
        columns = dataframe.columns
        data_for_seq = dataframe.values[start_i:end_i]

        def get_data(col):
            col_i = columns.get_loc(col)
            data = data_for_seq[:,col_i]
            len_data = len(data)
            zero_padded_data = np.zeros((self.seq_length, 1))
            zero_padded_data[:len_data,0] = data
            return zero_padded_data

        seq = Sequence(self.seq_length)
        seq.input = get_data('mains')
        seq.target = get_data('target')
        if building_name in self.vampire_power_per_building:
            seq.input = np.clip(
                seq.input-self.vampire_power_per_building[building_name],
                0, None)
        assert len(seq.input) == self.seq_length
        assert len(seq.target) == self.seq_length

        # Set mask
        seq.weights = np.ones((self.seq_length, 1), dtype=np.float32)
        n_zeros_to_pad = self.seq_length - len(data_for_seq)
        if n_zeros_to_pad > 0:
            seq.weights[-n_zeros_to_pad:, 0] = 0

        # Set metadata
        seq.metadata = {
            'seq_i': seq_i_for_building,
            'building_name': building_name,
            # this takes a lot of time:
            'start_date': dataframe.index[start_i],
            'end_date': dataframe.index[start_i+len(data_for_seq)-1]
        }

        return seq

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['data', '_num_seqs', 'rng']
