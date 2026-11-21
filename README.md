# From perception to imagery across the senses: EEG decoding of 3D shape in sighted, low-vision and blind adults

Analysis code for the study by **Zülal Tannur** (Department of Neuroscience and Cognitive Science, The University of Arizona; ORCID [0009-0002-0695-764X](https://orcid.org/0009-0002-0695-764X)).

64-channel EEG was recorded from 23 adults living in Türkiye (sighted, congenitally low-vision, congenitally blind and with later vision loss) while they perceived five 3D shapes (cone, cube, sphere, pyramid, cylinder) through hearing, touch and vision and then imagined them. Shape was decoded in people the model had never seen (leave-one-subject-out), and the role of visual experience in the format of imagery was examined.

## Repository layout

The analysis is organised in numbered stages. Each script runs from its own stage folder and reads from / writes to `results/<stage>/`, with input data under `data/` and `participants/` at the repository root.

| Stage | Content | Paper |
|---|---|---|
| `01`–`03` | Recording inspection, preprocessing (filtering, ICA, ICLabel), timing alignment, cleaning and epoching | Methods |
| `06_shape_decoding_main` | Features (spectral power, ERP, Riemannian) and the main analysis planned in advance | Results A1–A2 |
| `07_things_eeg2_pretraining` | EEG→CLIP encoder trained on THINGS-EEG2 and applied to the present data | A2, Figure S1 |
| `08`, `09` | Shape geometry (Chamfer distances) and superquadric reconstruction | Figure 1, A3–A4 |
| `10`, `11` | Additional geometric models and 3D-descriptor checks | Figure S6 |
| `12`, `13` | Active exploration and tactile hidden Markov model / TouchSim | A4, Figure S6 |
| `14_confirmation_test` | Confirmation test under a separate plan | A2 |
| `15`, `16` | Shape generation in every block; final hierarchical system | A1, Table 3, Figure 2 |
| `17_per_person_shapes` | Per-person shape tables | Figure S3 |
| `18_detailed_analyses` | Shape attributes, confusions and geometry | A3, Figure 3 |
| `19_congenital_low_vision_representation` | Training group × test group, group-label permutation | C1–C2, Table 4, Figure 4 |
| `20_time_frequency` | Time windows and frequency bands | C3, Table 5, Figure 6 |
| `21_visual_experience_and_regions` | Double dissociation, Mantel test, regions, searchlight | C4–C5, D1, Table 6, Figures 7–8, S2, S5 |
| `22_visual_experience_grouping` | Regrouping by visual experience | C5, Table 7 |
| `23_three_groups_cases_connectivity` | Group shapes, occipital alpha, touch–occipital connectivity | Figure 5, D1–D3 |
| `24_notes_and_signals` | Self-reported imagery strategy and decoded representation | E, Table 9 |
| `25_stability` | Stability of individual patterns | D5 |
| `26`–`28` | Responses to the word, measure ranking, word sound and meaning | D4–D5, A3, Table 8, Figures 9, S4 |
| `29_paper_figures` | All manuscript figures | Figures 1–9, S1–S6 |

## Requirements

Python 3.13 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Data availability

The de-identified dataset is openly available on Zenodo: https://doi.org/10.5281/zenodo.24415751 (raw recordings, cleaned continuous data, shape-labelled epochs, derived features, the results of every stage, the participant table and the experiment files). Unzip all archives into the repository root so that `data/`, `results/`, `participants/` and `experiment/` sit next to `src/`.

Participants appear only as anonymous labels P01–P23, identical to the labels used in the paper. Several analyses draw random subsets of participants from sorted lists; the dataset includes `participants/analysis_order.json`, which restores the participant order of the original analyses (see `src/participant_order.py`). With it, the scripts reproduce the reported numbers and figures exactly.

This code is archived on Zenodo: https://doi.org/10.5281/zenodo.24415753.

## Ethics

The study was approved by the Medical Research Evaluation Board (ATADEK) of Acıbadem Mehmet Ali Aydınlar University (decision no. 2025-13/509, 11 September 2025). All participants gave written informed consent.

## License

Code: MIT License (see `LICENSE`).

## Citation

Tannur, Z. (2026). *From perception to imagery across the senses: EEG decoding of 3D shape in sighted, low-vision and blind adults* [Preprint]. ScienceOpen.
