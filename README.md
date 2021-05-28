# Réseaux adverses génératifs (GAN)
Code source des cas pratiques réalisés dans le cadre de mon travail de Bachelor of Science HES⁠-⁠SO en Informatique de gestion (Business Information Technology) **Etude et application des réseaux adverses génératifs à la conversion de voix**  en 2021 à la [Haute école de gestion de Genève](https://www.hesge.ch/heg/).

Frédéric Charbonnier

## Cas pratiques
Les cas pratiques ont été développés en Python 3.7.0 avec TensorFlow 2.3.0.
### GAN 1D
Architecture simple basée sur une fonction à une dimension (1D) permettant de générer des points dans un espace donné.
### DCGAN avec MNIST
Intégration des réseaux de neurones convolutifs pour générer des lettres manuscrites en utilisant le jeu de données MNIST.
### CycleGAN pour la conversion de voix
Mise en place du concept CycleGAN-VC2 pour réaliser une conversion de voix.

## Références
- ALPHAROL, 2019. Voice-Conversion-CycleGAN2. GitHub [en ligne]. 4 novembre 2019. [Consulté le 26 mars 2021]. Disponible à : https://github.com/alpharol/Voice_Conversion_CycleGAN2
- BROWNLEE, Jason, 2019b. Generative Adversarial Networks with Python: Deep Learning Generative Models for Image Synthesis and Image Translation. Machine Learning Mastery, 2019
- GOODFELLOW, Ian J., POUGET-ABADIE, Jean, MIRZA, Mehdi, et al. Generative adversarial networks. arXiv preprint arXiv:1406.2661, 2014.
- KANEKO, Takuhiro, KAMEOKA, Hirokazu, TANAKA, Kou, et al. Cyclegan-vc2: Improved cyclegan-based non-parallel voice conversion. In : ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019. p. 6820-6824.
- KUN Ma, 2020. CycleGAN-VC2-PyTorch. GitHub [en ligne]. 27 novembre 2020. [Consulté le 26 mars 2021]. Disponible à : https://github.com/jackaduma/CycleGAN-VC2
- LORIA, Steven, 2013. WAV recording functionality using pyaudio. GitHub [en ligne]. 2 juin 2013. [Consulté le 16 mars 2021]. Disponible à : https://gist.github.com/sloria/5693955
- ZHU, Jun-Yan, PARK, Taesung, ISOLA, Phillip, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks. In : Proceedings of the IEEE international conference on computer vision. 2017. p. 2223-2232.

MIT License

Copyright (c) 2021 Frédéric Charbonnier