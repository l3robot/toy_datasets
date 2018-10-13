# Toy Datasets

This is a collection of toy datasets in my projects. Compatible with pytorch.

### SinMix: mixtures of Sinusoidals

The SinMix dataset is a dataset of mixture of sin. Each
sin in a mixture is form from an amplitude and a phase.
These parameters are randomly chosen.

**Hard-coded parameters (changed in the file for nows):**
    - `X_MIN`, `X_MAX`, `X_LENGTH`: 0, 10, 1000
    - `AMP_RANGE`: amplitude $$A$$ is chosen in $$[-4,4]$$
    - `PHASE_RANGE`: phase $$\\omega$$ is chosen in $$[-5,5]$$

**Command line parameters:**
    - `trainsize`: number of mixture in the trainset
    - `testsize`: number of mixture in the testset
    - `nb-sin`: number of sin per mixture

**Item of the dataset:**
    - index `0`: y on x
    - index `1`: [<list>amplitudes, <list>phases]

Hence, the dataset can be used to find the amplitudes
and the phases from the points or can be used in a
multi-task setting.