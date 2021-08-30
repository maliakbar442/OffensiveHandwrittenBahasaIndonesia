
from imagetrain import *
import logging
logging.basicConfig(level=logging.INFO)
from keras_preprocessing.image import img_to_array, load_img

def run():
    path = PATH
    train_generator, val_generator = load_data(path, split_ratio=0.25)
    model = define_model(val_generator.num_classes)


    try:
        img_path = ["D:/AIML/handwritten_data/test/5.jpg","D:/AIML/handwritten_data/test/download1.jpg"]

        s_path = img_path[0]
        image = load_img(s_path)
        logging.info("Testing image: " + s_path)

        input_arr = np.array([img_to_array(image)])
        c_pred = np.argmax(model.predict(input_arr))
        logging.info("prediction: class " + str(c_pred))

    except ValueError as e:
        logging.error("Value Error".format(e))
        logging.info("Make sure your image has dimension 400 X 700")

if __name__ == '__main__':
    checkpoint_filepath = 'model/imagebahasamodel.h5'
    if os.path.exists(checkpoint_filepath):
      logging.info("exist")
      run()
    else:
      imagetrain()
      logging.info("Check your model")

