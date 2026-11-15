# torch.profiler over the whole job: what a step is made of (GPU kernels by total device time).
import torch.profiler as _tp
_prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA]); _prof.__enter__()
def _done():
    _prof.__exit__(None, None, None)
    ka = _prof.key_averages()
    for key in ("device_time_total", "self_device_time_total", "cuda_time_total"):
        try:
            print("## PROFILE by", key); print(ka.table(sort_by=key, row_limit=45, max_name_column_width=70), flush=True); break
        except Exception as e:  # noqa: BLE001
            print("profile table failed", key, e)
after = [_done]
