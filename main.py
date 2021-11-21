import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def count_structs(image, struct):
    erosion = binary_erosion(image, struct) # эрозия оставляет пиксели только у подходящих фигур
    dilation = binary_dilation(erosion, struct) # наращиваем обратно нужные фигуры, чтобы попиксельно вычестить из основного изображения
    image -= dilation
    return label(dilation).max()

image = np.load("ps.npy").astype('uint')

structs = np.array([
            np.array([
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1]
            ]),
            np.array([
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]
            ]),
            np.array([
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1]
            ]),        
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ]),
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ])
        ], dtype=object)

total_obj = 0
for i in range(len(structs)):
    res = count_structs(image, structs[i])
    total_obj += res
    print(f'count objects = {res} for struct \n {structs[i]} \n')

print("total objects =", total_obj)
