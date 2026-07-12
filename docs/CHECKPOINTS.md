# Checkpoint registry

All hashes below are SHA-256 values. Do not rename a checkpoint without updating the local registry used for an experiment.

| Repository name | Source role | SHA-256 |
|---|---|---|
| `base_final.pth` | Original 17.97M-parameter base checkpoint | `e7f86ad8ba2baf11384e784346ef7f777cd0bd3abb2e4a33cc00809e6d16f374` |
| `best_distill_K5.pth` | Earlier five-step distillation checkpoint | `ca2a5a311ce869bb705b162e595b4e80a000426deecfb32a61f4d5700fcc4d1d` |
| `student_distilled_2291.pth` | Exact checkpoint used for the 22.91 dB paper result | `5d7bac873b0b915fe6c0679b103fea9afe25f70de3958cab8da3d8779d156a37` |
| `teacher_illum_ft.pth` | Transferred-teacher integrity protocol checkpoint | `a92c2988ced5cb0fb8e403fb143913d8fc2837fdb31dbeb1634aa3ca031697a2` |
| `fsd_seed3407_recovered.pth` | Frozen integrity FSD checkpoint | `a3775e843610d37e954063ae7f95274c63efcf9ec3527d39c0b51df4b342baa5` |
| `lolv1_distill_e20.pth` | LOL-v1 adaptation experiment | `b1f6477bb26675ec68d6f667bb855af8d51fd2bd12a81111be5a5629a217fbff` |

The exact paper checkpoint is `student_distilled_2291.pth`. The headline result must not be reproduced with `best_distill_K5.pth`, because it is a different checkpoint.

