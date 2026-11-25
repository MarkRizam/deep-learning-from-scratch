# AIR-004 experimental protocol
Educational reproduction; no novelty or peer-review claim. Fixed experiment configurations and seeds 11, 22, 33 are encoded in the runner. All configurations are reported, including unfavorable results. CPU float64 NumPy; PyTorch float64 reference only. Two identical runs compared with absolute tolerance 1e-9 and zero relative tolerance; platform differences may cause failures, which must be investigated rather than hidden.

- EXP-001: dense layer and cross-entropy gradients vs PyTorch, plus central difference of one weight (h=1e-5).
- EXP-002: sigmoid, tanh and ReLU derivatives on 1,000 normal inputs (SD 3). Fraction below .01 includes deliberately inactive ReLU units; zero is an illustrative ideal, not a competing algorithm.
- EXP-003: XOR truth table, MLP vs perceptron; training accuracy, not generalization.
- EXP-004: fixed 50 epochs, batch 30, learning rate .02; compare SGD, momentum and Adam final training loss to each method's first epoch loss, not initial loss or equivalent compute budget.
- EXP-005: normal weight initialization scales .01, .5, 2; 40 epochs Adam, same samples and order; final vs first epoch training loss.
- EXP-006: fixed synthetic classification data, 180 train and 60 held-out; 60 epochs, dropout 0/.3, L2 0/.01. All configurations evaluated, no winner selected or tuned against held-out data; L2 applies to every parameter including biases. The held-out set is a comparative evaluation set, not a final independent test after selecting a model.
- EXP-007: valid NCHW convolution forward and input/weight/bias gradients vs PyTorch, random upstream gradients.
- EXP-008: 160 synthetic 6x6 images, horizontal/vertical noisy lines; 120 train/40 held-out; CNN vs MLP, 35 epochs, Adam .01, batch 20. Architectures have different parameter counts; no compute-matched claim.

Report means and sample SD across three seeds. These are descriptive, not confidence intervals. No external datasets, sensitive data, paid APIs or GPU.
