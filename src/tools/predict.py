import


def predict(cfg, model, test_images, test_scribbles):
    model.eval()
    model_name = model.__class__.__name__

