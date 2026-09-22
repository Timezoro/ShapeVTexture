# Datasets

## Training Dataset

### STL-10
- **Source:** https://cs.stanford.edu/~acoates/stl10/
- **Created by:** Adam Coates, Honglak Lee, Andrew Y. Ng (2011)
- **Size:** 13,000 labelled 96×96 colour images — 5,000 train / 8,000 test
- **Classes:** 10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- **Resolution:** 96×96 — chosen over CIFAR-10 (32×32) so that shape and texture remain visually distinguishable in style-transferred cue-conflict images
- **License:** Freely available for research use; citation required.
- **Citation:**
  ```
  Coates, A., Lee, H., & Ng, A. Y. (2011). An Analysis of Single-Layer Networks in
  Unsupervised Feature Learning. AISTATS 2011.
  ```

---

## Evaluation Dataset

### Cue-Conflict Dataset (Self-Generated from CIFAR-10)
- **Source:** Generated from CIFAR-10 test images using neural style transfer
- **Method reference:** Geirhos et al. (2019) — https://github.com/rgeirhos/texture-vs-shape
- **Size:** 90 class pairs (10×9) × N images per pair
- **Purpose:** Images where shape comes from class A and texture comes from class B (A≠B). Used to measure whether the trained CNN predicts the shape class or the texture class — revealing shape bias vs. texture bias.
- **License:** Derived from CIFAR-10 (same terms apply — research use, citation required). No third-party dataset borrowed; fully self-contained within CIFAR-10.
- **Citation:**
  ```
  Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2019).
  ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.
  ICLR 2019. https://arxiv.org/abs/1811.12231
  ```

---

## Notes

- STL-10 is used for both **training** and **evaluation** — fully self-contained, no institutional ImageNet access required.
- STL-10's 96×96 resolution was chosen specifically because 32×32 (CIFAR-10) is too small for shape and texture to remain distinguishable after style transfer.
- Cue-conflict images are **generated**, not borrowed. Shape from class A + texture from class B via neural style transfer, covering all 90 ordered pairs.
- Stylized-ImageNet (SIN) and Geirhos' original cue-conflict stimuli are **not used** — SIN cannot be redistributed and their 16-class set does not align with STL-10.
