from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.io import imsave
import numpy as np
import pickle
import os 
from sklearn.metrics import mean_squared_error
import time
import random
import string
from PIL import Image


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

def generate_unique_filename():
    timestamp = str(int(time.time()))
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{timestamp}_{random_string}"

# Fonction de compression d'image
def compresser_image(image, k, output_file):
    # Appliquer la conversion en niveaux de gris
    image_gray = color.rgb2gray(image)

    # Appliquer la SVD
    U, S, VT = np.linalg.svd(image_gray, full_matrices=False)

    # Réduire la matrice avec les k premiers vecteurs et valeurs singulières
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]

    # Sauvegarder les matrices compressées dans un fichier
    with open(output_file, 'wb') as f:
        pickle.dump((U_k, S_k, VT_k, image_gray.shape), f)

    print(f"Image compressée sauvegardée dans {output_file}")

def compresser_image_color(image, k, output_file):
  
    

    if len(image.shape) == 3:  # Image RGB
        compressed_data = []
        for channel in range(3):  # Traiter chaque canal séparément (R, G, B)
            U, S, VT = np.linalg.svd(image[:, :, channel], full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
            VT_k = VT[:k, :]
            compressed_data.append((U_k, S_k, VT_k))
        image_shape = image.shape
        
    with open(output_file, 'wb') as f:
        pickle.dump((compressed_data, image_shape), f)

    print(f"Image compressée sauvegardée dans {output_file}")

# Fonction de décompression de l'image
def decompresser_image(output_file):
    with open(output_file, 'rb') as f:
        U_k, S_k, VT_k, original_shape = pickle.load(f)
    S_k_diag = np.diag(S_k)
    image_reconstructed = np.dot(U_k, np.dot(S_k_diag, VT_k))
    return image_reconstructed

def gain_compression(image_path, compressed_file, image_shape, dtype=np.float64):
    # Calculate the raw size of the original image
    taille_pixel = np.dtype(dtype).itemsize

    if len(image_shape) == 3:  # If the image is RGB
        taille_brute = image_shape[0] * image_shape[1] * image_shape[2] * taille_pixel
    elif len(image_shape) == 2:  # If the image is grayscale
        taille_brute = image_shape[0] * image_shape[1] * taille_pixel

    # Calculate the size of the compressed file
    taille_compressee = os.path.getsize(compressed_file)

    # Calculate the compression gain
    gain = (1 - taille_compressee / taille_brute) * 100
    return gain, taille_brute, taille_compressee

def erreur_quadratique_moyenne(image_originale, image_reconstruite):
    return np.mean((image_originale - image_reconstruite) ** 2)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imagegrey')
def imagegrey():
    return render_template('grayimage.html')

@app.route('/imagecolor')
def imagecolor():
    return render_template('colorimage.html')


@app.route('/imagegrey', methods=['POST'])
def compress():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return 'Aucune image sélectionnée'

    # Retrieve the uploaded image
    image_file = request.files['image']
    if image_file.filename == '':
        return 'Aucun fichier sélectionné'

    # Read the uploaded image
    image = io.imread(image_file)

    # Handle images with an alpha channel
    if image.shape[-1] == 4:  # RGBA image
        image = image[:, :, :3]  # Drop the alpha channel

    # Generate unique filenames for the images
    unique_filename = generate_unique_filename()
    original_image_path = f'static/{unique_filename}_original_image.png'
    reconstructed_image_path = f'static/{unique_filename}_reconstructed_image.png'
    output_file = f'static/{unique_filename}_compressed_image.IMSD'

    # Save the original image for display
    imsave(original_image_path, img_as_ubyte(color.rgb2gray(image)))

    # Get the compression factor (default to 50)
    try:
        k = int(request.form.get('k', '').strip() or 50)  # Use 50 if input is empty
    except ValueError:
        k = 50  # Fallback to 50 if the value cannot be converted to an integer
    if k < 0:
        k = 50

    # Compress the image
    compresser_image(image, k, output_file)

    # Decompress the image for display
    image_reconstructed = decompresser_image(output_file)

    # Normalize and convert the reconstructed image to uint8
    image_reconstructed = (image_reconstructed - image_reconstructed.min()) / (image_reconstructed.max() - image_reconstructed.min())
    image_reconstructed = img_as_ubyte(image_reconstructed)

    # Save the reconstructed image
    imsave(reconstructed_image_path, image_reconstructed)

    # Calculate compression gain
    gain, taille_brute, taille_compressee = gain_compression(
        image_file.filename, output_file, image.shape, dtype=np.float64
    )
    taille_brute = taille_brute / 1024
    taille_compressee = taille_compressee / 1024
    # Calculate mean squared error
    eqm = erreur_quadratique_moyenne(color.rgb2gray(image), image_reconstructed / 255.0)

    # Display results
    return render_template(
        'grayimage.html',
        original_image_path=original_image_path,
        reconstructed_image_path=reconstructed_image_path,
        gain=gain,
        taille_brute=taille_brute,
        taille_compressee=taille_compressee,
        eqm=eqm,
        k=k
    )



def decompresser_image_color(compressed_file):
  
    with open(compressed_file, 'rb') as f:
        compressed_data, image_shape = pickle.load(f)

    if len(compressed_data) == 3:  # Image RGB
        reconstructed_channels = []
        for U_k, S_k, VT_k in compressed_data:
            S_k_diag = np.diag(S_k)
            reconstructed_channel = np.dot(U_k, np.dot(S_k_diag, VT_k))
            reconstructed_channels.append(reconstructed_channel)
        image_reconstructed = np.stack(reconstructed_channels, axis=-1)
   
    return image_reconstructed

def erreur_quadratique_moyenne_color(image_originale, image_reconstruite):
    # Convert reconstructed image to grayscale
    image_reconstruite_gray = color.rgb2gray(image_reconstruite)
    return np.mean((image_originale - image_reconstruite_gray) ** 2)


def gain_compression_color(image_path, compressed_file, image_shape, dtype=np.float64):
   
    # Calculer la taille brute de l'image originale
    taille_pixel = np.dtype(dtype).itemsize
    
    if len(image_shape) == 3: 
        taille_brute = image_shape[0] * image_shape[1] * image_shape[2] * taille_pixel
         
    elif len(image_shape) == 2: 
        taille_brute = image_shape[0] * image_shape[1] * taille_pixel

    taille_compressee = os.path.getsize(compressed_file)

    # Calculer le gain de compression
    gain = (1 - taille_compressee / taille_brute) * 100
    return gain, taille_brute, taille_compressee

@app.route('/imagecolor', methods=['POST'])
def compresscolor():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return 'Aucune image sélectionnée'

    # Retrieve the uploaded image
    image_file = request.files['image']
    if image_file.filename == '':
        return 'Aucun fichier sélectionné'

    # Read the uploaded image
    image = io.imread(image_file)

    # Handle images with an alpha channel
    if image.shape[-1] == 4:  # RGBA image
        image = image[:, :, :3]  # Drop the alpha channel

    # Generate unique filenames for the images
    unique_filename = generate_unique_filename()
    original_image_path = f'static/{unique_filename}_original_image.png'
    reconstructed_image_path = f'static/{unique_filename}_reconstructed_image.png'
    output_file = f'static/{unique_filename}_compressed_image.IMSD'

    # Save the original image for display
    imsave(original_image_path, img_as_ubyte(image))

    # Get the compression factor (default to 50)
    try:
        k = int(request.form.get('k', '').strip() or 50)  # Use 50 if input is empty
    except ValueError:
        k = 50  # Fallback to 50 if the value cannot be converted to an integer
    if k < 0:
        k = 50

    # Compress the image
    compresser_image_color(image, k, output_file)

    # Decompress the image for display
    image_reconstructed = decompresser_image_color(output_file)

    # Normalize and convert the reconstructed image to uint8
    image_reconstructed = (image_reconstructed - image_reconstructed.min()) / (image_reconstructed.max() - image_reconstructed.min())
    image_reconstructed = img_as_ubyte(image_reconstructed)

    # Save the reconstructed image
    imsave(reconstructed_image_path, image_reconstructed)

    # Calculate compression gain
    gain, taille_brute, taille_compressee = gain_compression_color(
        image_file.filename, output_file, image.shape, dtype=np.float64
    )
    taille_brute = taille_brute / 1024
    taille_compressee = taille_compressee / 1024
    # Calculate mean squared error
    eqm = erreur_quadratique_moyenne_color(color.rgb2gray(image), image_reconstructed / 255.0)


    # Display results
    return render_template(
        'colorimage.html',
        original_image_path=original_image_path,
        reconstructed_image_path=reconstructed_image_path,
        gain=gain,
        taille_brute=taille_brute,
        taille_compressee=taille_compressee,
        eqm=eqm,
        k=k
    )
def svd_compression(image, k):
    U, S, VT = np.linalg.svd(image, full_matrices=False)
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    compressed_image = np.dot(U_k, np.dot(S_k, VT_k))
    return compressed_image

def calculate_svd_compression_gain(original_shape, k):
    m, n = original_shape
    original_elements = m * n
    compressed_elements = k * (m + n + 1)
    return compressed_elements / original_elements

# Sauvegarder et recharger une image en JPEG pour simuler la compression JPEG
def jpeg_compression(image, quality, temp_path="temp_image.jpg"):
    # Ensure the image is in the uint8 format
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    # Convert to PIL Image
    image_pil = Image.fromarray(image)

    # Save the image as a JPEG
    image_pil.save(temp_path, "JPEG", quality=quality)

    # Read back the compressed image
    compressed_image = Image.open(temp_path)

    # Calculate the compressed file size
    compressed_size = os.path.getsize(temp_path)

    return np.array(compressed_image), compressed_size




@app.route("/compare_compression")
def compare():
    return render_template('compare.html')


@app.route("/compare_compression",methods=["POST"])
def compare_compression():
    if 'image' not in request.files:
        return 'Aucune image sélectionnée'

    # Retrieve the uploaded image
    image_file = request.files['image']
    if image_file.filename == '':
        return 'Aucun fichier sélectionné'

    # Read the uploaded image
    image = io.imread(image_file)

    # Handle images with an alpha channel
    if image.shape[-1] == 4:  # RGBA image
        image = image[:, :, :3]  # Drop the alpha channel
    if len(image.shape) == 3:  # Check if the image has 3 dimensions (color image)
        image = color.rgb2gray(image)
    # Generate unique filenames for the images
    unique_filename = generate_unique_filename()
    original_image_path = f'static/{unique_filename}_original_image.png'
    output_file = f'static/{unique_filename}_compressed_image.IMSD'
    original_shape = image.shape
    original_size = original_shape[0] * original_shape[1]
    # Save the original image for display

    imsave(original_image_path, img_as_ubyte(image))

    k_values = [10, 20, 50]
    svd_images_path = []
    svd_mse = []
    svd_gain = []
    for k in k_values:
        compressed_image = svd_compression(image, k)
        reconstructed_image_path = f'static/{unique_filename}_reconstructed_image_{k}.png'
        image_reconstructed=compressed_image
        image_reconstructed = (image_reconstructed - image_reconstructed.min()) / (image_reconstructed.max() - image_reconstructed.min())
        image_reconstructed = img_as_ubyte(image_reconstructed)
        imsave(reconstructed_image_path, image_reconstructed)
        mse = mean_squared_error(image, compressed_image)
        gain = calculate_svd_compression_gain(original_shape, k)
        svd_images_path.append(reconstructed_image_path)
        svd_mse.append(mse)
        svd_gain.append(gain)

    # Compression JPEG
    jpeg_quality = [10, 50, 90]
    jpeg_mse = []
    jpeg_gain = []
    jpeg_images_path = []  # Store the paths to JPEG compressed images
    for quality in jpeg_quality:
        # Compress the image and retrieve the compressed size
        compressed_image, compressed_size = jpeg_compression(image, quality)
        
        # Generate a path to save the compressed image
        jpeg_image_path = f'static/{unique_filename}_compressed_image_quality_{quality}.jpg'
        
        # Save the compressed image
        imsave(jpeg_image_path, img_as_ubyte(compressed_image))
        
        # Calculate metrics
        mse = mean_squared_error(image, compressed_image)
        gain = compressed_size / original_size  # Ratio of compressed size to original size
        
        # Store paths and metrics
        jpeg_images_path.append(jpeg_image_path)
        jpeg_mse.append(mse)
        jpeg_gain.append(gain)

    return render_template(
        'compare.html',
        original_image_path=original_image_path,
        svd_images_path=svd_images_path,
        svd_mse=svd_mse,
        svd_gain=svd_gain,
        jpeg_images_path=jpeg_images_path,
        jpeg_mse=jpeg_mse,
        jpeg_gain=jpeg_gain,
        k_values=k_values,
        jpeg_quality=jpeg_quality,
        zip=zip  # Add zip to the context
    )

@app.route('/audio')
def audio():
    return render_template('audio.html')

    

  

if __name__ == "__main__":
    app.run(debug=True)
