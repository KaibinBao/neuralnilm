from __future__ import print_function, division
from copy import copy
import numpy as np

import nilmtk
from neuralnilm.utils import check_windows

import logging
logger = logging.getLogger(__name__)


def load_nilmtk_activations(appliances, filename, sample_period, windows,
                            on_power_thresholds=None,
                            min_on_durations=None,
                            min_off_durations=None,
                            sanity_check=1):
    """
    Parameters
    ----------
    appliances : list of strings
    filename : string
    sample_period : int
    windows : dict
        Structure example:
        {
            'train': {<building_i>: <window>},
            'unseen_activations_of_seen_appliances': {<building_i>: <window>},
            'unseen_appliances': {<building_i>: <window>}
        }

    Returns
    -------
    all_activations : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <appliance>: {
                 <building_name>: [<activations>]
        }}}
        Each activation is a pd.Series with DatetimeIndex and the following
        metadata attributes: building, appliance, fold.
    """
    logger.info("Loading NILMTK activations...")

    # check whether optional parameters are provided and if so, are of the same length as `appliances`
    # if not provided build a list of the same size as `appliances` and fill it with None entries.
    if (on_power_thresholds is not None) and (len(on_power_thresholds) != len(appliances)):
        raise ValueError("`on_power_thresholds` must have the same size as `appliances` ")
    elif on_power_thresholds is None:
        on_power_thresholds = [None for i in range(len(appliances))]

    if (min_on_durations is not None) and (len(min_on_durations) != len(appliances)):
        raise ValueError("`min_on_durations` must have the same size as `appliances` ")
    elif min_on_durations is None:
        min_on_durations = [None for i in range(len(appliances))]

    if (min_off_durations is not None) and (len(min_off_durations) != len(appliances)):
        raise ValueError("`min_off_durations` must have the same size as `appliances` ")
    elif min_off_durations is None:
        min_off_durations = [None for i in range(len(appliances))]

    if sanity_check:
        # Sanity check
        check_windows(windows)

    # Load dataset
    dataset = nilmtk.DataSet(filename)

    all_activations = {}
    for fold, buildings_and_windows in windows.items():
        logger.info(
            "Loading activations for fold {}.....".format(fold))         
        activations_for_fold = {}
        for building_i, window in buildings_and_windows.items():
            dataset.set_window(*window)
            elec = dataset.buildings[building_i].elec
            building_name = (
                dataset.metadata['name'] + '_building_{}'.format(building_i))
            for i, appliance in enumerate(appliances):
                logger.info(
                    "Loading {} for {}...".format(appliance, building_name))

                # Get meter for appliance
                try:
                    meter = elec[appliance]
                except KeyError as exception:
                    logger.info(building_name + " has no " + appliance +
                                ". Full exception: {}".format(exception))
                    continue

                # Get activations_for_fold and process them
                meter_activations = meter.get_activations(
                    sample_period=sample_period,
                    min_off_duration=min_off_durations[i],
                    min_on_duration=min_on_durations[i],
                    on_power_threshold=on_power_thresholds[i])
                meter_activations = [activation.astype(np.float32)
                                     for activation in meter_activations]

                # Attach metadata
                for activation in meter_activations:
                    activation._metadata = copy(activation._metadata)
                    activation._metadata.extend(
                        ["building", "appliance", "fold"])
                    activation.building = building_name
                    activation.appliance = appliance
                    activation.fold = fold

                # Save
                if meter_activations:
                    activations_for_fold.setdefault(
                        appliance, {})[building_name] = meter_activations
                logger.info(
                    "Loaded {} {} activations from {}."
                    .format(len(meter_activations), appliance, building_name))
        all_activations[fold] = activations_for_fold

    dataset.store.close()
    logger.info("Done loading NILMTK activations.")
    return all_activations
