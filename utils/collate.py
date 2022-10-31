import torch

__all__ = ['collate_fn', 'collate_fn_batch']


def collate_fn(data):
    pnames, imgs4d, m0s, mks, masks, init_ts, final_ts, ff, bf = zip(*data)

    init_ts = init_ts[0]
    final_ts = final_ts[0]
    times_fwd = torch.arange(init_ts, final_ts + 1, 1)
    times_bwd = torch.arange(final_ts, init_ts - 1, -1)

    imgs4d = torch.stack([x.float() for x in imgs4d], dim=0)
    m0s = torch.stack([x.float() for x in m0s], dim=0)
    mks = torch.stack([x.float() for x in mks], dim=0)
    ff = torch.stack([x.float() for x in ff], dim=0)
    bf = torch.stack([x.float() for x in bf], dim=0)
    imgs4d.unsqueeze_(1)
    m0s.unsqueeze_(1)
    mks.unsqueeze_(1)

    if masks[0] is not None:
        masks = torch.stack([x.float() for x in masks], dim=0)
        masks.unsqueeze_(1)
    else:
        masks = None

    return pnames, imgs4d, m0s, mks, masks, times_fwd, times_bwd, ff, bf


# Batch operations
def collate_fn_batch(data):
    pnames, imgs4d, m0s, mks, masks, init_ts, final_ts, ff, bf = zip(*data)

    times_fwd, times_bwd = collate_times(init_ts, final_ts)
    ff, bf, offsets = collate_optical_flow(ff, bf)

    imgs4d = torch.stack([x.float() for x in imgs4d], dim=0)
    m0s = torch.stack([x.float() for x in m0s], dim=0)
    mks = torch.stack([x.float() for x in mks], dim=0)
    imgs4d.unsqueeze_(1)
    m0s.unsqueeze_(1)
    mks.unsqueeze_(1)

    if masks[0] is not None:
        masks = torch.stack([x.float() for x in masks], dim=0)
        masks.unsqueeze_(1)
    else:
        masks = None

    return pnames, imgs4d, m0s, mks, masks, times_fwd, times_bwd, ff, bf, offsets


def collate_times(init_ts, final_ts):
    init_ts = torch.tensor(init_ts)
    final_ts = torch.tensor(final_ts)
    num_ts = (final_ts - init_ts).max().item()

    times_fwd = [init_ts]
    times_bwd = [final_ts]

    for _ in range(num_ts):
        next_time = times_fwd[-1] + 1
        times_fwd.append(torch.where(next_time > final_ts, final_ts, next_time))

        prev_time = times_bwd[-1] - 1
        times_bwd.append(torch.where(prev_time < init_ts, init_ts, prev_time))

    return(times_fwd, times_bwd)


def collate_optical_flow(off, ofb):
    BS = len(off)
    maxts_flow = max_ts(off)
    NZ, NY, NX, CH, NT = off[0].shape

    off_t = torch.zeros(size=(BS, NZ, NY, NX, CH, maxts_flow))
    ofb_t = torch.zeros(size=(BS, NZ, NY, NX, CH, maxts_flow))
    offsets = torch.zeros(BS, dtype=torch.int)

    for b in range(BS):
        # Prepare optical flow
        diff_of_ts = (int)(maxts_flow - off[b].shape[-1])
        offsets[b] = diff_of_ts
        if diff_of_ts != 0:
            zeros = torch.zeros(size=(NZ, NY, NX, 3, diff_of_ts))
            off_t[b] = torch.cat((off[b], zeros), dim=4)
            ofb_t[b] = torch.cat((ofb[b], zeros), dim=4)
        else:
            off_t[b] = off[b]
            ofb_t[b] = ofb[b]

    return (off_t, ofb_t, offsets)


def max_ts(ff):
    BS = len(ff)
    maxts = 0
    for b in range(BS):
        t = ff[b].shape[-1]
        if t > maxts:
            maxts = t
    return maxts
