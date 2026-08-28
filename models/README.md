# Supplied model checkpoints

These files came from the submitted research archive and are retained as PyTorch state
dictionaries. They were renamed for clarity; their contents were not changed.

| Current filename | Original filename |
| --- | --- |
| `encoder_source_cic2018.pt` | `encoder_source_srcCIC_2018.pickle_2021_12_08_09_21_midlay10_b1024_lr0.001_acSigmoid_optSGD_pzFalse_fil_OK_17.31.pt` |
| `classifier_source_cic2018.pt` | `classifier_source_srcCIC_2018.pickle_2021_12_08_09_21_midlay10_b1024_lr0.001_acSigmoid_optSGD_pzFalse_fil_OK_17.31.pt` |
| `encoder_source_unsw_nb15.pt` | `encoder_source_srcUNSW_NB15.pickle_2021_12_07_21_57_midlay10_b1024_lr0.001_acSigmoid_optSGD_pzFalse_fil_OK_61.94.pt` |
| `classifier_source_unsw_nb15.pt` | `classifier_source_srcUNSW_NB15.pickle_2021_12_07_21_57_midlay10_b1024_lr0.001_acSigmoid_optSGD_pzFalse_fil_OK_61.94.pt` |

The encoder trained with CIC-2018 as the source should be used for the CIC-2018 to UNSW-NB15
direction. The UNSW-NB15 source encoder should be used for the reverse direction.

Only load model files obtained from a trusted source. The project loader restricts deserialisation
to weights, but keeping a checksum record remains good practice. Verify these files with:

```bash
sha256sum -c models/SHA256SUMS
```

