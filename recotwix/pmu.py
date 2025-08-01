import torch

def get_actual_trigger_timing(twix, trigger_method='ECG1'):
    """
    Get actual trigger timing from the twix data. Remove trigger not seen during acquisition.

    Parameters:
    - twix: The twix data structure.
    - trigger_method: The method used to trigger the acquisition.

    Returns:
    - Array of trigger timings.
    """
    assert 'pmu' in twix and any(twix['pmu'].trigger[trigger_method]), \
        f"No PMU data found for trigger method '{trigger_method}'."
    
    # Get trigger lock time
    try:
        trigger_lock_time = twix['hdr']['MeasYaps']['sPhysioImaging']['lTriggerLockTime']*1e-6 # convert to seconds
    except:
        Warning('Fail to read trigger lock time. Assuming no lock time')
        trigger_lock_time = 0
    
    pmu = twix['pmu']
    start_time = twix['mdb'][0].mdh.TimeStamp
    mask = pmu.trigger[trigger_method]>0
    trigger_timing = pmu.timestamp_trigger[trigger_method][mask]
    trigger_timing = (trigger_timing - start_time)  * 2.5e-3 # convert to seconds
    
    actual_trigger_timing = [ trigger_timing[0] ] 
    last_trigger = trigger_timing[0]
    for trigger in trigger_timing[1:]:
        if (trigger-last_trigger) >= trigger_lock_time:
            actual_trigger_timing.append(trigger)
            last_trigger = trigger
    return torch.tensor(actual_trigger_timing)