import os
import time
import numpy as np
from multiprocessing import Process, cpu_count, Manager
import argparse
import psutil
from pathlib import Path
from PIL import Image


# Configuración general
IMAGE_SHAPE = (224, 224)

#Desviación y media obtenidas del conjunto ImageNet
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
INPUT_DIR = "/home/jiu/Documents/School IPN/Sexto Semester/Computo Paralelo/Segundo Dpto/Proyecto - Final de Dpto/ImageNet/ILSVRC/Data/CLS-LOC"


def read_image_paths(directory):
    return list(Path(directory).rglob("*.JPEG"))


def generate_synthetic_image():
    img = np.random.randint(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
    return img

def preprocess_image(img):
    try:
        img = Image.open(img).convert('RGB')
        img = img.resize(IMAGE_SHAPE)
        img_np = np.array(img).astype(np.float32)
        img_np = (img_np - MEAN) / STD
        img_np = img_np.transpose(2, 0, 1)
        return img_np
    except Exception as e:
        print(f"Error con la imagen: {img} - {e}")
        return None

def worker(proc_id, num_images_proc, result_dict, list_paths_image):
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    start_time = time.time()
    
    for _ in list_paths_image[(proc_id-1)*num_images_proc:((proc_id-1)*num_images_proc) + num_images_proc]:
        __ = preprocess_image(_)

    end_time = time.time()
    cpu_usage = process.cpu_percent(interval=None)

    result_dict[proc_id] = {
        'tiempo': end_time - start_time,
        'cpu': cpu_usage,
        'procesadas': num_images_proc
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_procs', type=int, default=cpu_count(), help='Número de procesos')
    args = parser.parse_args()
    NUM_IMAGES = len(read_image_paths(INPUT_DIR))
    IMAGE_PATHS = read_image_paths(INPUT_DIR)
    imgs_per_proc = NUM_IMAGES // args.num_procs
    print(f"Total imágenes: {NUM_IMAGES}")
    print(f"Procesos: {args.num_procs} ({imgs_per_proc} imágenes por proceso)\n")

    # === Simulación secuencial ===
    print("Ejecutando simulación secuencial (1 proceso)...")
    t0 = time.time()
    for _ in IMAGE_PATHS:
        __ = preprocess_image(_)
    t1 = time.time()
    tiempo_secuencial = t1 - t0
    print(f"Tiempo secuencial: {tiempo_secuencial:.2f} s\n")

    # === Procesamiento paralelo ===
    print("Iniciando procesamiento paralelo...")
    manager = Manager()
    result_dict = manager.dict()
    processes = []

    start_parallel = time.time()
    for i in range(args.num_procs):
        start = i * imgs_per_proc
        end = NUM_IMAGES if i == args.num_procs  else (imgs_per_proc * (i+1))
        p = Process(target=worker, args=(i+1, end-start, result_dict, IMAGE_PATHS[start:end]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_parallel = time.time()

    tiempo_paralelo = end_parallel - start_parallel
    speedup = tiempo_secuencial / tiempo_paralelo
    eficiencia = speedup / args.num_procs

    print("\nResultados por proceso:")
    for pid, info in result_dict.items():
        print(f"  [P{pid}] Tiempo: {info['tiempo']:.2f}s | CPU: {info['cpu']}% | Procesadas: {info['procesadas']}")

    print("\nMétricas de Evaluación:")
    print(f"Tiempo secuencial: {tiempo_secuencial:.2f} s")
    print(f"Tiempo paralelo: {tiempo_paralelo:.2f} s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Eficiencia: {eficiencia*100:.2f}%\n")

if __name__ == "__main__":
    main()
