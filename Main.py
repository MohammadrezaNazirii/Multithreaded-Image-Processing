import cv2
import numpy as np
import threading
from scipy.ndimage import convolve
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def number_threads(image):
    n_thread = max(9, len(image) // 50)
    return n_thread

# Convert RGB to Grayscale
def rgb_to_grayscale(image):
    r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    grayscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale_image.astype(np.uint8)

# Divide image into sub-matrices with overlap
def divide_image(image, num_threads):
    height, width = image.shape[:2]
    chunk_height = height // num_threads
    sub_matrices = []
    for i in range(num_threads):
        start_row = max(i * chunk_height - 1, 0)
        end_row = min((i + 1) * chunk_height + 1, height)
        sub_matrix = image[start_row:end_row, :]
        sub_matrices.append((start_row, sub_matrix))
    return sub_matrices

# Apply Sobel filter for edge detection
def sobel_edge_detection(sub_matrix, result_matrix, start_row, lock):
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows, cols = sub_matrix.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = np.sum(gx_kernel * sub_matrix[i - 1: i + 2, j - 1: j + 2])
            gy = np.sum(gy_kernel * sub_matrix[i - 1: i + 2, j - 1: j + 2])
            gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
            lock.acquire()
            result_matrix[start_row + i, j] = 255 - gradient_magnitude  # Invert the gradient magnitude
            lock.release()

# Define the Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply Laplacian filter for edge detection
def laplacian_edge_detection(sub_matrix, result_matrix, start_row, lock):
    # Apply the Laplacian kernel using convolution
    laplacian_sub_matrix = convolve(sub_matrix.astype(np.float64), laplacian_kernel)
    
    # Acquire the lock, update the result matrix, and release the lock
    lock.acquire()
    result_matrix[start_row:start_row + sub_matrix.shape[0], :] = np.uint8(np.absolute(laplacian_sub_matrix))
    lock.release()

# Apply Noise filter
def add_noise(sub_matrix, result_matrix, start_row, lock):
    noise = np.random.normal(0, 1, sub_matrix.shape).astype(np.uint8)
    sub_matrix = cv2.add(sub_matrix, noise)
    lock.acquire()
    result_matrix[start_row:start_row + sub_matrix.shape[0], :] = sub_matrix
    lock.release()

# Function to generate a Gaussian kernel
def get_gaussian_kernel(size, sigma):
    if sigma == 0:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# Apply Gaussian Differential filter
def gaussian_differential(sub_matrix, result_matrix, start_row, lock):
    # Apply Gaussian blur using custom Gaussian kernel
#     kernel = get_gaussian_kernel(3, 0)
#     print(kernel)
    kernel = np.array( [[0.05711826, 0.12475775, 0.05711826],
                        [0.12475775, 0.27249597, 0.12475775],
                        [0.05711826, 0.12475775, 0.05711826]])
    blurred_sub_matrix = convolve(sub_matrix, kernel)

    # Define Sobel kernels
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    rows, cols = blurred_sub_matrix.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = np.sum(gx_kernel * blurred_sub_matrix[i - 1: i + 2, j - 1: j + 2])
            gy = np.sum(gy_kernel * blurred_sub_matrix[i - 1: i + 2, j - 1: j + 2])
            gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
            lock.acquire()
            result_matrix[start_row + i, j] = 255 - gradient_magnitude  # Invert the gradient magnitude
            lock.release()

# Reassemble image from sub-matrices
def reassemble_image(result_matrix, height, width):
    result_image = result_matrix[1: height - 1, :]
    return result_image

# Save image to disk
def save_image(image, path):
    cv2.imwrite(path, image)
    
# Calculate the mean of the resulting matrix
def calculate_mean(matrix):
    mean_value = np.mean(matrix)
    print(f"Mean value: {mean_value}")
    return mean_value

# Apply thresholding based on the mean value
def apply_thresholding(matrix, mean_value):
    thresholded_matrix = np.where(matrix > mean_value, 0, 255)
    return thresholded_matrix.astype(np.uint8)

def process_image(image_path, filter_type='Sobel', num_threads=9):
    image = load_image(image_path)
    num_threads = number_threads(image)
    grayscale_image = rgb_to_grayscale(image)
    sub_matrices = divide_image(grayscale_image, num_threads)

    height, width = grayscale_image.shape
    result_matrix = np.zeros((height, width), dtype=np.uint8)
    threads = []
    lock = threading.Lock()

    if filter_type == 'Sobel' or filter_type == 'Sobel with Threshold':
        filter_function = sobel_edge_detection
    elif filter_type == 'Laplacian':
        filter_function = laplacian_edge_detection
    elif filter_type == 'Noise':
        filter_function = add_noise
    elif filter_type == 'Gaussian Differential':
        filter_function = gaussian_differential
    
    for start_row, sub_matrix in sub_matrices:
        thread = threading.Thread(
            target=filter_function,
            args=(sub_matrix, result_matrix, start_row, lock),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    result_image = reassemble_image(result_matrix, height, width)
    
    # Calculate the mean of the resulting matrix
    mean_value = calculate_mean(result_image)

    # Apply thresholding
    if filter_type == 'Sobel with Threshold':
        thresholded_image = apply_thresholding(result_image, mean_value)
    else:
        thresholded_image = result_image
    
    return thresholded_image

# Tkinter GUI
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("600x700")

        self.file_path = tk.StringVar()
        self.filter_type = tk.StringVar(value='Sobel')

        self.create_widgets()

    def create_widgets(self):
        # File selection
        tk.Label(self.root, text="Select Image:").pack(pady=10)
        tk.Entry(self.root, textvariable=self.file_path, width=50).pack(pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_file).pack(pady=5)

        # Filter selection
        tk.Label(self.root, text="Select Filter:").pack(pady=10)
        filter_options = ['Sobel', 'Sobel with Threshold', 'Laplacian', 'Noise', 'Gaussian Differential']
        tk.OptionMenu(self.root, self.filter_type, *filter_options).pack(pady=5)

        # Process button
        tk.Button(self.root, text="Process Image", command=self.process_image).pack(pady=20)

        # Display area
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.file_path.set(file_path)

    def process_image(self):
        image_path = self.file_path.get()
        filter_type = self.filter_type.get()

        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return

        try:
            result_image = process_image(image_path, filter_type=filter_type)
            save_path = "output_image.jpg"
            save_image(result_image, save_path)

            # Display result image
            result_pil_image = Image.fromarray(result_image)
            result_pil_image.thumbnail((400, 400))
            result_tk_image = ImageTk.PhotoImage(result_pil_image)

            self.image_label.config(image=result_tk_image)
            self.image_label.image = result_tk_image
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
