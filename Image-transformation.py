#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:24:27 2023

TD09 Image

@author: enzo duval
"""





import numpy as np
import matplotlib.pyplot as plt
import copy
import math

im = np.zeros((500, 500, 3), dtype=np.uint8)
im[:, :, :] = 255
plt.imshow( im)
plt.show()
plt.imsave("Monochrome.jpg", im)

# Oeuvre_perso_ED.jpg

def neg(image):
    """
    Affiche la négative de l'image fournie à l'aide de la fonction imshow() de la bibliothèque Matplotlib.

    Args:
    image (str): Le chemin d'accès du fichier de l'image à inverser.

    Returns:
    None.
    """
    imaege = plt.imread(image, format= 'jpg')
    dimension = imaege.shape
    imaege = imaege.copy()
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            p = imaege[x,y]
            imaege[x,y] = [255 - p[0], 255 - p[1] , 255 - p[2] ]
    
    plt.imshow(imaege)
    plt.title( "l'image negative ")
    return plt.show()
            


neg("route.jpg")
neg("carnaval Binche.jpg")
neg("lion marchant.jpg")
neg("paysage neige.jpg")



def gris(image):
    """
  Transforme une image en couleur en une image en nuance de gris.

  Args:
  image (str): Le chemin d'accès du fichier de l'image à transformer.

  Returns:
  numpy.ndarray: Un tableau Numpy représentant l'image en nuance de gris.
  """
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            t = img[x,y]       
            n= (1/3*t[0]+1/3*t[1]+1/3*t[2])//1
            img[x,y] = [n ,n ,n ]

    plt.imshow(img)
    plt.title( "l'image en nuance de gris ")
    plt.show()
    return img        
    
    
gris("route.jpg")  
gris("carnaval Binche.jpg")
gris("lion marchant.jpg")
gris("paysage neige.jpg")





def gris_rouge(image):
    """
  Transforme une image en couleur en une image en nuance de gris en mettant l'accent sur le canal rouge.

  Args:
  image (str): Le chemin d'accès du fichier de l'image à transformer.

  Returns:
  None: La fonction affiche l'image en nuance de gris avec un accent sur le canal rouge à l'aide de Matplotlib.
  """
    
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            t = img[x,y]
            n= (0.5*t[0]-0.25*t[1]+0.25*t[2])//1
            img[x,y] = [n ,n ,n ]
            
    plt.imshow(img)
    plt.title( "l'image en nuance de gris avec de l'importance sur le rouge ")
    return plt.show()
 
gris_rouge("route.jpg")
gris("route.jpg")



def NB(image):
    """
   Transforme une image en couleur en une image en noir et blanc en utilisant un seuil de 127 pour la conversion.

   Args:
   image (str): Le chemin d'accès du fichier de l'image à transformer.

   Returns:
   None: La fonction affiche l'image en noir et blanc à l'aide de Matplotlib.
   """
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            t = img[x,y]       
            n= (1/3*t[0]+1/3*t[1]+1/3*t[2])//1
            if n < 255/2:
                p=0
            if n > 255/2:
                p=255
            img[x,y] = [p ,p ,p ]
            
            
    plt.imshow(img)
    plt.title( "l'image en noir et blanc ")
    return plt.show()
    
    
 
NB("route.jpg")   
 

def sepia(image) :
    """
   Applique un effet sépia à l'image spécifiée par le chemin d'accès 'image'.
   
   Parameters:
   image (str): Le chemin d'accès de l'image en format jpg.
   
   Returns:
   img (array): Une représentation de l'image en format tableau numpy, avec l'effet sépia appliqué.
   """
    sepia = [94,38,18]
    img=gris(image)
    dimension = img.shape
    img = img.copy()
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            a = img[x,y][0]
            if a < 120:
                for i in range (3):
                    img[x,y][i]= int((a/120) * sepia[i])
            if a >=120 :
                for i in range (3):
                    img[x,y][i] = int (255 - ((255 - sepia[i])* (120/a) ))
                    
    plt.imshow(img)
    plt.title( " l'image en sépia  ")
    plt.show()
    return img
                    
                
    
    
sepia("route.jpg")    
    
    
    





def Contraste (image):
    """Applique un effet de contraste à l'image donnée en paramètre.

    Parameters:
    image (str): Le nom du fichier image à modifier.
    
    Returns:
    None
    """
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            t = img[x,y] 
            for i in range (3):
                if t[i]<30:
                    t[i]=0
                if t [i]>200:
                    t[i]=255
            img[x,y] = [t[0] ,t[1] ,t[2] ]
            
            
    plt.imshow(img)
    plt.title( "le contraste de l'image ")
    return plt.show()



   
Contraste ("carnaval Binche.jpg")   






def Bordure (image):
    """
   Trace une bordure rouge autour de l'image.
   
   Args:
   image: une image en format .jpg
   
   Returns:
   Affiche l'image avec une bordure rouge.
   """
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    a, b = 0, 10
    c, d = dimension[0]-10, dimension[0]

    union = list(range(a, b )) + list(range(c, d ))
    union = list(set(union))
    
    a2, b2 = 0, 10
    c2, d2 = dimension[1]-10, dimension[1]
    print (union)

    union2 = list(range(a2, b2 )) + list(range(c2, d2 ))
    union2 = list(set(union2))
    print (union2)
    for x in range (dimension[0]):
        for y in union2 :
            img[x,y] = [255 ,0 ,0 ]
    for x in union:
        for y in range ( dimension[1]) :
            img[x,y] = [255 ,0 ,0 ]
   
    plt.imshow(img)
    plt.title( "bordure rouge")
    return plt.show()

Bordure ("lion marchant.jpg")
          
            

    







def Sysmétrie (image):
    """
    Effectue une symétrie verticale de l'image donnée en entrée.
    
    :param image: un fichier image au format JPG
    :type image: str
    
    :return: une représentation graphique de l'image symétrique
    """
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    a=dimension [1]//2
    for x in range (dimension [0]):
        for y in range (a) :
            
            img[x,y] = img[x,y] ^ img[x,dimension[1]-y-1]
            img[x,dimension[1]-y-1] = img[x,y] ^ img[x,dimension[1]-y-1]
            img[x,y] = img[x,y] ^ img[x,dimension[1]-y-1]
    
    plt.imshow(img)
    plt.title( "la symétrie ")
    return plt.show()
    
    
    

Sysmétrie ("route.jpg")





def Upside_Down (image):
    """
    Effectue une symétrie horizontale de l'image donnée en entrée.
    
    :param image: un fichier image au format JPG
    :type image: str
    
    :return: une représentation graphique de l'image symétrique
    """
    img = plt.imread(image, format= 'jpg')
    dimension = img.shape
    img = img.copy()
    a=dimension [0]//2
    for x in range (a):
        for y in range (dimension [1]) :
            img[x,y] = img[x,y] ^ img[dimension[0]-x-1,y]
            img[dimension[0]-x-1,y] = img[x,y] ^ img[dimension[0]-x-1,y]
            img[x,y] = img[x,y] ^ img[dimension[0]-x-1,y]
    
    plt.imshow(img)
    plt.title( "La symétrie bonus")
    return plt.show()



Upside_Down ("route.jpg")



img = plt.imread("lion marchant.jpg", format= 'jpg')
dimension1 = img.shape
img = img.copy()
print (img [100,100])



img2 = plt.imread("paysage neige.jpg", format= 'jpg')
dimension2 = img2.shape
img2 = img2.copy()

if dimension1[0] == dimension2[0] and dimension1[1] == dimension2[1] :
    print ("les images sont de même dimension")



def Fusion(im1,im2):
    """
    Prend deux images (au format .jpg) en entrée et effectue une fusion de l'image 2 sur l'image 1 
    en remplaçant les pixels verts de l'image 1 par les pixels de l'image 2.
    
    Args:
        im1 (str): le nom de fichier de la première image à fusionner (format .jpg)
        im2 (str): le nom de fichier de la deuxième image à fusionner (format .jpg)
        
    Returns:
        Affiche l'image fusionnée.
    """
    img1 = plt.imread(im1, format= 'jpg')
    dimension1 = img1.shape
    img1 = img1.copy()
    
    img2 = plt.imread(im2, format= 'jpg')
    dimension2 = img2.shape
    img2 = img2.copy()
    d=img1[500,500]
    print (500,500 )
    # print(dimension1[0])
    for x in range (dimension1[0]):
        for y in range (dimension1[1]) :
            t = img1[x,y]
            if img1[x,y][1] >150 and t[1] < 255 and t[2] < 100 and t[0] < 140 and t[0] > 40:
                img1[x,y] = img2[x,y]
    plt.imshow(img1)
    plt.title( "l'incrustation du lyon dans la foret")
    return plt.show()      
    
    
    


Fusion("lion marchant.jpg","paysage neige.jpg")



print("en effet l'image est pas parfaite j'aurais donc proposer de faire un test (que j'ai effecuter) sur la variable bleu et la variable rouge de ces pixels vert pour eviter de faire disparaitre notre lion")






















    

Moyenne = np.array ([[1,1,1],[1,1,1],[1,1,1]])
Moyenne = Moyenne/9
print (Moyenne)

Gausienne = np.array ([[1,2,1],[2,4,2],[1,2,1]])
Gausienne = Gausienne/16



    







def Conv(M1,M2):
    """
    Calcule la convolution entre deux matrices M1 et M2.

    Args:
    - M1: une matrice 3x3
    - M2: une matrice 3x3

    Returns:
    - La valeur de la convolution entre M1 et M2, c'est-à-dire la somme des produits
      élément par élément de M1 et M2.

    Exemple:
    >>> M1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> M2 = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    >>> Conv(M1, M2)
    0
    """

    s=0
    for i in range (3):
        for j in range (3):
            s=s+M1[i][j]*M2[i][j]
    return s 
    
print (Conv(Moyenne,Gausienne))


def Floutage(opérateur, image):
    """
   Applique l'opérateur de convolution donné à l'image spécifiée pour produire une version floutée de celle-ci.
   
   Paramètres:
   - opérateur : un tableau de coefficients de convolution
   - image : le nom de fichier d'une image au format jpg
   
   Retour:
   - Cette fonction ne retourne rien, elle affiche simplement l'image floutée.
   """
    img1 = plt.imread(image, format= 'jpg')
    dimension1 = img1.shape
    img1 = img1.copy()
    
    imnew = plt.imread("route.jpg", format= 'jpg')
    imnew = imnew.copy()
    for x in range (dimension1[0]):
        for y in range (dimension1[1]):
            imnew[x,y]= [255,255,255] 
    for x in range (1,dimension1[0]-1):
        for y in range (1,dimension1[1]-1):
            for i in range (3):
                
                matrice = np.array ([[img1[x-1,y-1][i],img1[x-1,y][i],img1[x-1,y+1][i]],[img1[x,y-1][i],img1[x,y][i],img1[x,y+1][i]],[img1[x+1,y-1][i],img1[x+1,y][i],img1[x+1,y+1][i]]])
                imnew[x,y][i]= int(Conv(matrice,opérateur))
                
                
    plt.imshow(imnew)
    plt.title( "image flouter")
    return plt.show() 
    
    

Floutage(Gausienne, "route.jpg")

laplacien = np.array ([[1,0,1],[0,-4,0],[1,0,1]])

sfr = np.array ([[1,1,1],[1,-8,1],[1,1,1]])

def contour(image):
    """Fonction qui détecte les contours d'une image en niveaux de gris.

    Args:
        image (str): Le nom de l'image à traiter.

    Returns:
        None: La fonction affiche directement l'image des contours détectés.
    """
    seuil = 17
    img=gris(image)
    dimension = img.shape
    img = img.copy()
    
    imnew = plt.imread("route.jpg", format= 'jpg')
    imnew = imnew.copy()
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            imnew[x,y]= [255,255,255]
            
    for x in range (1,dimension[0]-1):
        for y in range (1,dimension[1]-1):
            for i in range (1):
                
                matrice = np.array ([[img[x-1,y-1][i],img[x-1,y][i],img[x-1,y+1][i]],[img[x,y-1][i],img[x,y][i],img[x,y+1][i]],[img[x+1,y-1][i],img[x+1,y][i],img[x+1,y+1][i]]])
                a= abs(int(Conv(matrice,laplacien)))
                if a > seuil :
                    imnew[x,y]=[255,255,255]
                else :
                    imnew[x,y]=[0,0,0]
    
    
    plt.imshow(imnew)
    plt.title( "detection de contour laplacien")
    plt.show()
    
    
    imnew = plt.imread("route.jpg", format= 'jpg')
    imnew = imnew.copy()
    for x in range (1,dimension[0]-1):
        for y in range (1,dimension[1]-1):
            for i in range (1):
                
                matrice = np.array ([[img[x-1,y-1][i],img[x-1,y][i],img[x-1,y+1][i]],[img[x,y-1][i],img[x,y][i],img[x,y+1][i]],[img[x+1,y-1][i],img[x+1,y][i],img[x+1,y+1][i]]])
                a= abs(int(Conv(matrice,sfr)))
                if a > seuil :
                    imnew[x,y]=[255,255,255]
                else :
                    imnew[x,y]=[0,0,0]   
                    
    plt.imshow(imnew)
    plt.title( "detection de contour sfr")
    return plt.show()
                
    
    
 
    
    
    
    



contour("route.jpg")




print("les dimensions de l'image réduite sera le quotient de la division euclidienne de Nl par n et Nc par n autrement dit : Nl//n et Nc//n")


def Red_simple(n, image):
    """
    Réduit la taille d'une image en conservant uniquement un pixel tous les n pixels dans chaque direction.
    
    Args:
        n (int): facteur de réduction. Doit être un entier positif.
        image (str): chemin vers l'image à réduire.
    
    Returns:
        None: la fonction affiche simplement l'image réduite.
    """
    img1 = plt.imread(image)
    hauteur, taille, _ = img1.shape

    new_hauteur = hauteur // n
    new_taille = taille // n

    new_image = np.zeros((new_hauteur, new_taille, 3), dtype=np.uint8)

    for i in range(new_hauteur):
        for j in range(new_taille):
            new_image[i, j] = img1[i*n, j*n]

    plt.imshow(new_image)
    plt.title("image reduite ")
    plt.show()

Red_simple(10, "route.jpg")



print("plus le facteur n de reduction est grand et plus l'image est flou, pas nette, elle sera donc pixelisé ")

# faire q2










def Moyenne_paquets(image): 
    """
    Calcule la moyenne des couleurs de l'image en parcourant tous les pixels.

    Args:
        image (str): Le chemin d'accès à l'image.

    Returns:
        list: Une liste de trois entiers représentant la moyenne des couleurs
            de l'image dans l'espace RGB.

    Example:
        >>> Moyenne_paquets("chemin/vers/image.jpg")
        [128, 64, 32]
    """
    S=0
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    for x in range (dimension[0]):
        for y in range (dimension[0]):
            S=(1/(dimension[0]**2))*img1[x,y]+S
    S=[int(S[0]),int(S[1]),int(S[2])]
    return S
  
"""

"""          
def Moyenne_paquetsbis(matrice):
    """
    Calcule la moyenne des couleurs de l'image en parcourant tous les pixels.

    Args:
        image (str): Le chemin d'accès à l'image.

    Returns:
        list: Une liste de trois entiers représentant la moyenne des couleurs
            de l'image dans l'espace RGB.

    Example:
        >>> Moyenne_paquets("chemin/vers/image.jpg")
        [128, 64, 32]
    """
    S=0
    img1 = matrice
    dimension = matrice.shape
    for x in range (dimension[0]):
        for y in range (dimension[0]):
             S=(1/(dimension[0]**2))*img1[x,y]+S
    S=[int(S[0]),int(S[1]),int(S[2])]
    return S               
    
    


print(Moyenne_paquets("route.jpg"))





def Mat_red(n,image,l,c):
    """
    Renvoie la sous-matrice de l'image `image` centrée en `(l, c)` et de dimension `n x n`.
    Si `(l, c)` n'est pas le centre d'une sous-matrice de dimension `n x n`, renvoie la sous-matrice
    centrée en la position la plus proche possible.

    Args:
        n (int): la dimension de la sous-matrice carrée à extraire
        image (str): le chemin vers le fichier image
        l (int): l'indice de la ligne du pixel central de la sous-matrice
        c (int): l'indice de la colonne du pixel central de la sous-matrice

    Returns:
        numpy.ndarray: une sous-matrice carrée de dimension `n x n` extraite de l'image `image` centrée en `(l, c)`
    """
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    
    
    l1=l%n
    c1=c%n
    ldep=l-l1+1
    cdep=c-c1+1
    
    matrice = img1[ldep:ldep+n, cdep:cdep+n]
    return matrice

def Mat_redbis(n,matrice,l,c):
    """
    Renvoie la sous-matrice de `matrice` centrée en `(l, c)` et de dimension `n x n`.
    Si `(l, c)` n'est pas le centre d'une sous-matrice de dimension `n x n`, renvoie la sous-matrice
    centrée en la position la plus proche possible.

    Args:
        n (int): la dimension de la sous-matrice carrée à extraire
        matrice (numpy.ndarray): une matrice 2D
        l (int): l'indice de la ligne du pixel central de la sous-matrice
        c (int): l'indice de la colonne du pixel central de la sous-matrice

    Returns:
        numpy.ndarray: une sous-matrice carrée de dimension `n x n` extraite de `matrice` centrée en `(l, c)`
    """





    img1=matrice
    dimension = img1.shape
    img1 = img1.copy()
    
    
    l1=l%n
    c1=c%n
    ldep=l-l1+1
    cdep=c-c1+1
    
    matrice = img1[ldep:ldep+n, cdep:cdep+n]
    return matrice




Mat_red(10,"route.jpg",100,40)



def Red_paquets(n,image):
    """
    Réduit la taille d'une image en divisant celle-ci en paquets de taille n x n pixels,
    puis en prenant la moyenne de chaque paquet pour obtenir un nouveau paquet de taille 1 x 1 pixel.
    Les nouveaux paquets ainsi obtenus sont utilisés pour créer une nouvelle image réduite.

    Args:
    - n : int : la taille des paquets carrés qui vont être formés dans l'image originale
    - image : str : le nom du fichier image à réduire
    
    Returns:
    - None : la fonction affiche l'image réduite, mais ne retourne pas de valeur
    """
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    
    hauteur, taille, _ = img1.shape

    new_hauteur = hauteur // n
    new_taille = taille // n

    new_image = np.zeros((new_hauteur, new_taille, 3), dtype=np.uint8)
    
    for x in range (1,dimension[0]-dimension[0]%n,n):
        for y in range (1,dimension[1]-dimension[1]%n,n):
            new_image[int((x+n-1)/n)-1,int((y+n-1)/n)-1]= Moyenne_paquetsbis(Mat_redbis(n,img1,x,y))
    
    
    plt.imshow(new_image)
    plt.title("image reduite ")
    plt.show()
            
            
    

Red_paquets(4,"route.jpg")



print("les dimensions de l'image agrandie sera Nl * n +1 et Nc * n +1, ")

print("les 4 pixel initiaux les plus proche du pixel [l,c] dans l'image final sera : ")
print("")
print("[l-l%n,c-c%n],[l-l%n+n,c-c%n],[l-l%n,c-c%n+n],[l-l%n+n,c-c%n+n]")    



def Ag_voisins(n,image):
    """
   Agrandit une image en utilisant la méthode des voisins.

   Args:
       n (int): Le facteur d'agrandissement.
       image (str): Le chemin d'accès à l'image.

   Returns:
       None

   Raises:
       FileNotFoundError: Si le chemin d'accès à l'image est invalide.

   """
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    
    hauteur, taille, _ = img1.shape

    new_hauteur = (hauteur-1) * n +1
    new_taille = (taille-1)* n +1

    new_image = np.zeros((new_hauteur, new_taille, 3), dtype=np.uint8)
    for x in range ((dimension[0]-1)*(n)+1):
        for y in range ((dimension[1]-1)*(n)+1):
            if x%n< n/2 and y%n < n/2 :
                new_image[x,y]= img1[x//n,(y//n)]
            if x%n < n/2 and y%n >= n/2 :
                new_image[x,y]= img1[x//n,(y//n)+1]
            if x%n >= n/2 and y%n >= n/2 :
                new_image[x,y]= img1[x//n+1,(y//n)+1]
            if x%n >= n/2 and y%n < n/2 :
                new_image[x,y]= img1[x//n+1,(y//n)]
                
                
    plt.imshow(new_image)
    plt.title("image agrandie ")
    plt.show()
    

            
                
            
Ag_voisins(6,"route.jpg")   
    
    
    






def Ag_bil(n,image):
    """Agrandit une image en utilisant la méthode du bilinéaire.

    Arguments :
    - n : le facteur d'agrandissement de l'image (un entier positif)
    - image : le nom du fichier image à agrandir (au format jpg)
    
    Retour :
    - Affiche l'image agrandie
    
    """
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    
    hauteur, taille, _ = img1.shape

    new_hauteur = hauteur * n -2
    new_taille = taille * n -2
    
    new_image = np.zeros((new_hauteur, new_taille, 3), dtype=np.uint8)
    for x in range ((dimension[0]-1)*(n)-1):
        for y in range ((dimension[1]-1)*(n)-1):
            new_image[x,y]=  (1-((y%n)/n))*((((x%n)/n)*(img1[x//n+1,y//n])) + ((1-((x%n))/n)*(img1[x//n,y//n]))) + (((y%n)/n))*((((x%n)/n)*(img1[x//n+1,y//n+1])) + (((1-((x%n))/n))*(img1[x//n,y//n+1])))         
    plt.imshow(new_image)
    plt.title("image agrandie ")
    plt.show()    
     
    
    
    
Ag_bil(4,"route.jpg") 
    


def Coord_verteur(P,C):
    """
    Calcule le vecteur entre deux points.

    Arguments:
    P -- liste des coordonnées du premier point [x,y]
    C -- liste des coordonnées du deuxième point [x,y]

    Return:
    Coord -- liste des coordonnées du vecteur entre les deux points [x,y]
    """
    Coord = []
    for i in range (2):
        Coord.append(P[i]-C[i])
    return Coord


def Rot_point(alpha,C,P):
    """
    Effectue une rotation d'un point autour d'un centre de rotation.

    Arguments:
    alpha -- angle de rotation en radians
    C -- liste des coordonnées du centre de rotation [x,y]
    P -- liste des coordonnées du point à tourner [x,y]

    Return:
    PP -- liste des coordonnées du point tourné [x,y]
    """
    CP= Coord_verteur(P,C)
    # print (CP)
    matricealpha = np.array ([[math.cos(alpha),-math.sin(alpha)],[math.sin(alpha),math.cos(alpha)]])
    # print (matricealpha[0][0])
    CPP = [matricealpha[0][0]*CP[0]+matricealpha[0][1]*CP[1],matricealpha[1][0]*CP[0]+matricealpha[1][1]*CP[1]]
    PP = [int(CPP[0]+C[0]),int(CPP[1]+C[1])]
    return PP



print(Rot_point(30,[4,2],[2,5]))




def Rot(image,alpha,centre):
    """Fonction qui permet de faire une rotation d'image d'un angle donné autour d'un centre de rotation donné.
   
    Args:
       image (str): Chemin vers l'image à traiter.
       alpha (float): Angle de rotation (en radians).
       centre (list): Liste contenant les coordonnées du centre de rotation de l'image.
       
    Returns:
       None.
    """
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    
    hauteur, taille, _ = img1.shape

    new_hauteur = hauteur
    new_taille = taille 
    
    new_image = np.ones((new_hauteur, new_taille, 3), dtype=np.uint8)
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            D=Rot_point(alpha,centre,[x,y])
            if D[0]>0 and D[0]<dimension[0] and D[1]>0 and D[1]<dimension[1] :
                new_image[D[0],D[1]] = img1[x,y]
                
    
    plt.imshow(new_image)
    plt.title("image tourner ")
    plt.show()
    
    
    
            
    
    

Rot("route.jpg",math.pi /4 ,[300,300])

print("si le centre est trop proche des bordure et l'angle assez important alors on ne voit quasiment plus l'image  ( elle sors du cadre ) ")

def Rot_point_ind(alpha,C,P):
    """
    Applique une rotation de alpha radians au point P autour du centre C et renvoie les coordonnées du point tourné.

    Args:
        alpha (float): Angle de rotation en radians.
        C (list[int]): Centre de rotation, sous forme d'une liste [x, y] des coordonnées x et y.
        P (list[int]): Point à tourner, sous forme d'une liste [x, y] des coordonnées x et y.

    Returns:
        list[int]: Coordonnées du point tourné, sous forme d'une liste [x, y] des coordonnées x et y.
    """
    CP= Coord_verteur(P,C)
    # print (CP)
    matricealpha = np.array ([[math.cos(alpha),math.sin(alpha)],[-math.sin(alpha),math.cos(alpha)]])
    # print (matricealpha[0][0])
    CPP = [matricealpha[0][0]*CP[0]+matricealpha[0][1]*CP[1],matricealpha[1][0]*CP[0]+matricealpha[1][1]*CP[1]]
    PP = [int(CPP[0]+C[0]),int(CPP[1]+C[1])]
    return PP



def Rot_ind(image,alpha,centre):
    """
    Applique une rotation de alpha radians à l'image située à l'emplacement donné par 'image' autour du centre donné par 'centre' et affiche l'image tournée.

    Args:
        image (str): Emplacement de l'image à tourner.
        alpha (float): Angle de rotation en radians.
        centre (list[int]): Centre de rotation, sous forme d'une liste [x, y] des coordonnées x et y.
    """
    
    img1 = plt.imread(image, format= 'jpg')
    dimension = img1.shape
    img1 = img1.copy()
    
    hauteur, taille, _ = img1.shape

    new_hauteur = hauteur
    new_taille = taille 
    
    new_image = np.ones((new_hauteur, new_taille, 3), dtype=np.uint8)
    
    for x in range (dimension[0]):
        for y in range (dimension[1]):
            D=Rot_point_ind(alpha,centre,[x,y])
            if D[0]>0 and D[0]<dimension[0] and D[1]>0 and D[1]<dimension[1] :
                new_image[x,y] = img1[D[0],D[1]]
                
    
    plt.imshow(new_image)
    plt.title("image tourner inversement")
    plt.show()
    
    
    
    
Rot_ind("route.jpg",math.pi /4 ,[300,300])   
    
    
    
    
    
    
    
print("l'algorithme de rotation indirection effectue une rotation de l'image beaucoup plus net que la rotation direct")



    
