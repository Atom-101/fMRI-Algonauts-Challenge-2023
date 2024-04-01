import numpy as np
import pickle
import os

subj = "01"
in_dims = {'01': 39548, '02': 39548, '03': 39548, '04': 39548, '05': 39548, '06': 39198, '07': 39548, '08': 39511}

arrays = ["floc-bodies_challenge_space",
          "floc-faces_challenge_space",
          "floc-places_challenge_space",
          "floc-words_challenge_space",
          "prf-visualrois_challenge_space",
          "streams_challenge_space"]
path = f"/fsx/proj-fmri/shared/algonauts_data/subj{subj}/roi_masks"

subj_masks = {}
all_masks = []
lh_offset = 0
for hem in ("lh", "rh"):
    for array in arrays:
        array_np = np.load(f"{path}/{hem}.{array}.npy")
        if hem=="lh":
            lh_offset = len(array_np)

        for unique_idx in np.unique(array_np)[1:]:
            mask = np.zeros(in_dims[subj], dtype=bool)
            mask_ids = np.where(array_np==unique_idx)[0]
            if hem=="lh":
                mask[mask_ids] = True
            else:
                mask[lh_offset:][mask_ids] = True
            subj_masks[f"{hem}.{array}_{unique_idx}"] = mask
            all_masks.append(mask)

all_masks = np.stack(all_masks, axis=0).sum(0).astype(bool)
if not all(all_masks):
    mask = np.zeros(in_dims[subj], dtype=bool)
    mask[~all_masks] = True
    subj_masks["background"] = mask

import pdb; pdb.set_trace()
pickle.dump(subj_masks, open("/fsx/proj-fmri/shared/algonauts_wds2/subj{subj}_masks.pkl", "wb"))
os.chmod("/fsx/proj-fmri/shared/algonauts_wds2/subj{subj}_masks.pkl", 0o777)

        
        

