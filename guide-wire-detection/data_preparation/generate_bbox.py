import cv2, os, glob, asyncio, tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Iterable, Tuple


def get_sample_feed_dict(data_path: str, gt_subfolder_name: str) -> Dict[str, Iterable[str]]:

    sample_subfolders = [
        os.path.join(data_path, path) for path in os.listdir(data_path)
    ]
    sample_path_dict = {
        path : glob.glob(os.path.join(path, gt_subfolder_name, "*.png"), recursive=True)
        for path in sample_subfolders if ("Test_data" not in path or "Train_data" not in path)
    }
    return sample_path_dict


def draw_bbox(
        gt_path: str, 
        dx: int=-15, 
        dy: int=-15, 
        dw: int=40, 
        dh: int=40, 
        save_bbox: bool=False, 
        save_dir: Optional[str]=None,
        center_xy: bool=True, 
        scale_bbox: bool=True) -> Tuple[np.ndarray, Iterable[float]]:
    
    sample_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    H, W = sample_img.shape
    min_pixel = sample_img.min()
    max_pixel = sample_img.max()
    threshold_pixel = np.ceil(max_pixel / 2)

    #1. apply thresholding on the gray image to create a binary image
    #2. Find the contour and use the first one to draw the bounding rectangles
    #3. Adjust the coordinates and dimensions of the retangle
    #4. Overlay the rectangle on the image
    _, thresh = cv2.threshold(sample_img, threshold_pixel, max_pixel, min_pixel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    x += dx
    y += dy
    w += dw
    h += dh
    bbox = (x, y, w, h)

    bbox_img = cv2.rectangle(sample_img, (x, y), (x+w, y+h), (255), 5)

    if center_xy:
        x = x + (w/2)
        y = y + (h/2)

    if scale_bbox:
        x = x/W
        y = y/H
        w = w/W
        h = h/H

    bbox = (x, y, w, h)
    if save_bbox:
        bbox_filename = f"{os.path.split(gt_path)[-1].split('.')[0]}.txt"
        save_dir = os.path.join(save_dir, "BBox")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, bbox_filename), "w") as f:
            f.write(" ".join(["0"] + list(map(lambda x : str(x), bbox))))
        f.close()

    return bbox_img, bbox


async def draw_bbox_coroutine(
        gt_path: str, 
        semaphore: asyncio.Semaphore,
        dx: int=-15, 
        dy: int=-15, 
        dw: int=40, 
        dh: int=40, 
        save_bbox: bool=False, 
        save_dir: Optional[str]=None):
    
    running_loop = asyncio.get_running_loop()
    func = lambda : draw_bbox(
        gt_path=gt_path, 
        dx=dx, 
        dy=dy, 
        dw=dw, 
        dh=dh, 
        save_bbox=save_bbox, 
        save_dir=save_dir)
    
    async with semaphore:
        await running_loop.run_in_executor(None, func)


async def main(
        sample_path_dict: Dict, 
        n_concurrency: int=10,
        dx: int=-15, 
        dy: int=-15, 
        dw: int=40, 
        dh: int=40):
    
    tasks = []
    semaphore = asyncio.Semaphore(n_concurrency)
    
    for sample_dir in sample_path_dict:
        sample_gt_files = sample_path_dict[sample_dir]
        for gt_path in sample_gt_files:
            tasks.append(draw_bbox_coroutine(
                gt_path=gt_path, 
                semaphore=semaphore, 
                dx=dx, 
                dy=dy, 
                dw=dw, 
                dh=dh, 
                save_bbox=True, 
                save_dir=sample_dir)
            )
    
    [await _ for _ in tqdm.tqdm(asyncio.as_completed(tasks))]

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Annotate Train data
        data_path = "data/Train_data/"
        gt_subfolder_name = "gt"
        sample_feed_dict = get_sample_feed_dict(data_path, gt_subfolder_name)
        logging.info("Annotating Training data:")
        asyncio.run(main(sample_feed_dict, n_concurrency=30))

        # Annotate Validation data
        data_path = "data/Validation_data/"
        gt_subfolder_name = "Label"
        sample_feed_dict = get_sample_feed_dict(data_path, gt_subfolder_name)
        logging.info("Annotating Validation data:")
        asyncio.run(main(sample_feed_dict, n_concurrency=30))

        # Annotate Test data
        data_path = "data/Test_data/"
        gt_subfolder_name = "Label"
        sample_feed_dict = get_sample_feed_dict(data_path, gt_subfolder_name)
        logging.info("Annotating Testing data:")
        asyncio.run(main(sample_feed_dict, n_concurrency=30))

    except Exception as e:
        logging.error(e)