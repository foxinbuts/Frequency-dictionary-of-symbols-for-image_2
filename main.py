#%%
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from pathlib import Path
from skimage.morphology import binary_dilation

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1: -1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) -1

def count_vlines(region):
    return np.all(region.image, axis = 0).sum()

def count_lgr_vlines(region):
    x = region.image.mean(axis=0) == 1
    return np.sum(x[:len(x)//2]) > np.sum(x[len(x)//2:])

def recognize(region):
    if np.all(region.image):
        return "-"
    else:  # 9 symbols
        holes = count_holes(region)
        if holes ==2: # 8 or B
            _, cx = region.centroid_local
            cx /= region.image.shape[1]
            if cx < 0.44:
                return "B"
            return "8"
        elif holes == 1: # A, 0, D, P
            _, cy = region.centroid_local
            cy /= region.image.shape[0]
            _, cx = region.centroid_local
            cx /= region.image.shape[1]
            if cy < 0.56 and count_vlines(region) > 0:
                return "P"
            if cx > 0.56 and count_vlines(region) > 0 and cy < 0.05:
                return "D"
            if count_vlines(region) > 0:
                return "0"
            else:
                return "A"
        else: # 1, *, /, X, W
            if count_vlines(region) >= 3:
                return "1"
            else:
                if region.eccentricity <= 0.4:
                    return "*"
                # /, X, W
                inv_image = ~region.image
                inv_image = binary_dilation(inv_image, np.ones((3, 3)))
                labeled = label(inv_image, connectivity=1)
                match np.max(labeled):
                    case 2: return "/"
                    case 4: return "X"
                    case _: return "W"
                    
    return "#"

symbols = plt.imread(Path(__file__).parent / "symbols.png")
gray = symbols[:, :, :-1].mean(axis = 2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)
print(len(regions))

result = {}
out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)
plt.figure()
for i, region in enumerate(regions):
    print(f"{i+1}/{len(regions)}")
    symbol = recognize(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(symbol)
    plt.imshow(region.image)
    plt.savefig(out_path / f"{i:03d}.png")

print(result)
#%%