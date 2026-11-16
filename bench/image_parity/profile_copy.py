import torch, time
for pin in (False, True):
    a = torch.empty(2 * 2**30, dtype=torch.uint8, pin_memory=pin)
    a.fill_(1)
    for _ in range(2):
        torch.cuda.synchronize(); t = time.time(); g = a.to("cuda", non_blocking=pin); torch.cuda.synchronize()
        print("pinned" if pin else "pageable", "2 GB H2D", round(2 / (time.time() - t), 1), "GB/s", flush=True)
        del g
