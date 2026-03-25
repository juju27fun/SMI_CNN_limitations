# Filter Combination Study — Report

## Objective

Determine which combination of the 3 bandpass filters yields the best
classification accuracy for the Conv1D particle classifier.

## Filters

| ID | Name | Type | Band | Stage |
|----|------|------|------|-------|
| F1 | Generation filter | Butterworth (scipy) | 7–80 kHz | Data generation |
| F2 | Training filter | FFT bandpass (torch) | 5–100 kHz | Training transform |
| F3 | Notebook filter | FFT bandpass (torch) | 8–40 kHz | Training transform |

## Results

| Rank | Combination | Filters active | Test Accuracy | Test Loss | Best Val Accuracy |
|------|------------|----------------|---------------|-----------|-------------------|
| 1 | F2 | F2 | 0.9768 **best** | 0.0502 | 0.9917 |
| 2 | F1_F2 | F1, F2 | 0.9735 | 0.0441 | 0.9917 |
| 3 | F1 | F1 | 0.9669 | 0.0874 | 0.9876 |
| 4 | F3 | F3 | 0.9669 | 0.0788 | 0.9876 |
| 5 | F2_F3 | F2, F3 | 0.9669 | 0.0946 | 0.9793 |
| 6 | F1_F2_F3 | F1, F2, F3 | 0.9669 | 0.0711 | 0.9793 |
| 7 | no_filter | (none) | 0.9570 | 0.1891 | 0.9834 |
| 8 | F1_F3 | F1, F3 | 0.9570 | 0.1436 | 0.9876 |

## Conclusion

The best filter combination is **F2** (filters: F2) with a test accuracy of **0.9768**.

### Observations

- **0 filter(s)**: average test accuracy = 0.9570
- **1 filter(s)**: average test accuracy = 0.9702
- **2 filter(s)**: average test accuracy = 0.9658
- **3 filter(s)**: average test accuracy = 0.9669

---
*Generated automatically by `run_filter_combinations.py`*
