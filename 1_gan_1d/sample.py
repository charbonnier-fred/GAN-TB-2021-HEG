from numpy import concatenate, hstack, ones, zeros
from numpy.random import permutation, rand, randn 

# Fonction carré
def fcarre(x):
    return x * x

# Génère n vrais échantillons
def generate_real_samples(n):
    # Génère un vecteur d'entrées entre -0.5 et 0.5
    X = rand(n) - 0.5
    # Génère le vecteur de sortie de fcarre(x)
    Y = fcarre(X)
    # Transforme les vecteurs en matrice 1 dimension
    X = X.reshape(n, 1)
    Y = Y.reshape(n, 1)
    # Concatène les 2 matrice 1 dimension
    # pour obtenir une matrice à 2 dimension
    XY = hstack((X, Y))
    # Crée une matrice 1 dimension remplie de 1 pour
    # indiquer qu'il s'agit de vrais échantillons
    Z = ones((n, 1))
    return XY, Z

"""
# Génère n faux échantillons
def generate_fake_samples(n):
    # Génère un vecteur entre -0.5 et 0.5
    x1 = rand(n) - 0.5
    # Génère un vecteur entre -0.5 et 0.5
    x2 = rand(n) - 0.5
    # Transforme les vecteurs en matrice 1 dimension
    x1 = x1.reshape(n, 1)
    x2 = x2.reshape(n, 1)
    # Concatène les 2 matrice 1 dimension
    # pour obtenir une matrice à 2 dimension
    x = hstack((x1, x2))
    # Crée une matrice 1 dimension remplie de 0 pour
    # indiquer qu'il s'agit de faux échantillons
    y = zeros((n, 1))
    return x, y


# Génère n faux échantillons à l'aide du générateur
def generate_fake_samples(generator, latent_dim, n):
    # Génère les points de l'espace latent
    x_input = generate_latent_points(latent_dim[0], n)
    # Génère les échantillons à l'aide du générateur
    x = generator.predict(x_input)
    # Crée une matrice 1 dimension remplie de 0 pour
    # indiquer qu'il s'agit de faux échantillons
    y = zeros((n, 1))
    return x, y


# Génère n/2 vrais échantillons et n/2 faux échantillons
def generate_samples(n):
    half_n = int(n / 2)
    # Génère des vrais échantillons
    x_real, y_real = generate_real_samples(half_n)
    # Génère les faux échantillons
    x_fake, y_fake = generate_fake_samples(half_n)
    # Concatène les vrais et faux échantillons
    x = concatenate((x_real, x_fake))
    y = concatenate((y_real, y_fake))
    # Mélange x et y dans le même ordre 
    assert len(x) == len(y)
    p = permutation(len(y))
    return x[p], y[p]

# Générer les points de l'espace latent
def generate_latent_points(latent_dim, n):
    # Génère les points à l'aide de la fonction randn() utilisant la distribution gaussienne
    x_input = randn(latent_dim * n)
    # Remodèle dans une matrice à plusieurs dimensions
    x_input = x_input.reshape(n, latent_dim)
    return x_input
"""