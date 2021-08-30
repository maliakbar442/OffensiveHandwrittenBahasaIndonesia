import logging

from tensorflow.python.keras.callbacks import ModelCheckpoint
from model import *
from config.configure import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def imagetrain():
    batch_size = 4
    epochs = 10
    checkpoint_filepath = 'model/imagebahasamodel.h5'
    metrics = "val_accuracy"
    path = PATH
    print(path)

    retrain_model = True
    logging.basicConfig(level=logging.INFO)

    train_generator, val_generator = load_data(path, split_ratio=0.25)
    log_info(train_generator)
    #
    model = define_model(val_generator.num_classes)
    logging.info(val_generator.num_classes)
    if not retrain_model:
        model.load_weights(checkpoint_filepath)
    model.summary()
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 verbose=1,
                                 monitor=metrics,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')

    if retrain_model:
        hist = model.fit(train_generator,
                     steps_per_epoch=None,
                     epochs=epochs,
                     verbose=1,
                     validation_data=val_generator,
                     validation_steps=None,
                     callbacks=[checkpoint])


    result = model.evaluate(val_generator, return_dict=True, batch_size=1, verbose=1)
    logging.info("Evaluation", result)

    train_generator, val_generator = load_data(path, split_ratio=0.25, shuffle=False)
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    logging.info('Confusion Matrix \n')
    logging.info(confusion_matrix(val_generator.classes, y_pred))
    logging.info('\n\n accuracy score \n')
    logging.info(accuracy_score(val_generator.classes, y_pred))
    logging.info('\n \n Classification Report \n')

    classes = train_generator.class_indices
    label_map = np.array(list(classes.items()))
    label = label_map[:, 0].tolist()
    target_names = label
    logging.info(classification_report(val_generator.classes, y_pred, target_names=target_names))

