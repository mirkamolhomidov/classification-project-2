import fastai
from fastai.vision.all import *
from ipywidgets import widgets
import cv2 as cv
path = Path('./OIDv4_ToolKit/OID/train')
if __name__ == '__main__':
    transports = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224)
    )

    dls = transports.dataloaders(path)
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(4)

    image = cv.imread('bolide.jpg')
    img = PILImage.create(image)
    pred, pred_id, probs = learn.predict(img)
    print("Prediction = ", pred)
    print("Probablity = ", probs[pred_id])

    learn.export('classify-mohirdev.pkl')
