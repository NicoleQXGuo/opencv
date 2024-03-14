import numpy as np
import os
from keras_segmentation.models import all_models
from keras_segmentation.predict import predict_multiple, predict, evaluate
os.environ['KERAS_BACKEND'] = 'tensorflow'


tr_im = "./dataset/images_train"
tr_an = "./dataset/annotations_train"
te_im = "./dataset/images_test"
te_an = "./dataset/annotations_test"


def train_model():
    model_name = "resnet50_unet"
    h = 320
    w = 384
    n_c = 2
    check_path = './logs/vgg_unet_1'

    m = all_models.model_from_name[model_name](n_c, input_height=h, input_width=w)

    m.train(
        train_images=tr_im,
        train_annotations=tr_an,
        steps_per_epoch=15,
        epochs=200,
        checkpoints_path=check_path,
        augmentation_name='aug_geometric', do_augment=True
    )

    # m.predict_segmentation(np.zeros((h, w, 3))).shape

    predict_multiple(
        inp_dir=te_im, checkpoints_path=check_path, out_dir="./logs")
    predict_multiple(inps=[np.zeros((h, w, 3))] * 3,
                     checkpoints_path=check_path, out_dir="./logs")

    ev = m.evaluate_segmentation(inp_images_dir=te_im, annotations_dir=te_an)
    assert ev['frequency_weighted_IU'] > 0.01
    print(ev)
    o = predict(inp=np.zeros((h, w, 3)), checkpoints_path=check_path)

    o = predict(inp=np.zeros((h, w, 3)), checkpoints_path=check_path,
                overlay_img=True, class_names=['nn'] * n_c, show_legends=True)
    print("pr")

    o.shape

    ev = evaluate(inp_images_dir=te_im, annotations_dir=te_an, checkpoints_path=check_path)
    assert ev['frequency_weighted_IU'] > 0.01


if __name__ == '__main__':
    train_model()
