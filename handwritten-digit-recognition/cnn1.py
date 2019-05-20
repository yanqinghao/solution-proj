from mlstudiosdk.modules.components.component import LComponent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from mlstudiosdk.modules.algo.data import Table,Domain
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.exceptions.exception_base import Error
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable

class CNN(LComponent):
    category = 'Algorithm'
    name = 'CNN'
    title = 'CNN Algorithm'
    # inputs = [('Data_IN',Table,'set_data')]
    # outputs = [('Data_OUT', Table)]
    inputs = [("Train Data", Table, "set_traindata"),
              ("Test Data", Table, "set_testdata")]
    outputs = [("News", Table),
               ("Predictions", Table),
               ("Evaluation Results", Results),
               ("Columns", list),
               ("Metas", list),
               ("Metric Score", MetricFrame),
               ("Metric", MetricType)]

    def __init__(self):
        super().__init__()
        self.data = None

    # def set_data(self,data):
    #     self.data = data
    def set_traindata(self, data):
        self.train_data = data

    def set_testdata(self, data):
        self.test_data = data

    def label_str2number(self,data,label_str):
        format = lambda x: label_str .index(x)
        data[self.label_name] = data[self.label_name].map(format)
        return data

    def run(self):
        # get label & sentence columns name
        self.label_name = self.train_data.domain.class_var.name
        # if len(self.train_data.domain.variables) - len(self.train_data.domain.class_vars) > 1 \
        #         #         #         or len(self.train_data.domain.attributes) > 1:
        #         #         #     raise Error("too many features")
        # self.sentence_name = self.train_data.domain.attributes[0].name
        #get test data
        test_data = table2df(self.test_data)
        try:
            test_data.iloc[:, :-len(self.test_data.domain.class_vars)] = \
                test_data.iloc[:, :-len(self.test_data.domain.class_vars)].astype('float')
        except Exception as e:
            print(e)
            raise Error('data must be numeric')

        test_data = test_data.dropna(subset=[self.label_name])
        if len(test_data) != len(self.test_data):
            raise Error("test data missing label")
        train_data = table2df(self.train_data)
        try:
            train_data.iloc[:, :-len(self.train_data.domain.class_vars)] = \
                train_data.iloc[:, :-len(self.train_data.domain.class_vars)].astype('float')
        except Exception as e:
            print(e)
            raise Error('data must be numeric')
        if len(test_data.columns) != len(train_data.columns):
            raise Error('train data and test data must match')
        train_data = train_data.dropna(subset=[self.label_name])

        train_x = train_data.drop([self.label_name],axis=1)
        train_y = train_data[self.label_name]
        train_x = train_x.values.reshape(-1, 28, 28, 1) / 255.0
        train_y = pd.get_dummies(train_y)
        self.train_y_label = list(train_y.columns)
        train_data = self.label_str2number(train_data,self.train_y_label)
        test_data = self.label_str2number(test_data, self.train_y_label)
        self.test_size = 0.1
        # self.random_state=42
        self.random_state = None
        X_train, X_val, Y_train, Y_val = \
            train_test_split(train_x, train_y.values, test_size=self.test_size, random_state=self.random_state)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y_label), activation="softmax"))

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                    min_lr=0.00001)
        self.epochs = 3  # Turn epochs to 30 to get 0.9967 accuracy
        batch_size = 86
        # Without data augmentation i obtained an accuracy of 0.98114
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(X_train)
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                      epochs=self.epochs, validation_data=(X_val, Y_val),
                                      verbose=2, steps_per_epoch=train_x.shape[0] // batch_size
                                      , callbacks=[learning_rate_reduction])
        val_score = history.history["val_acc"][self.epochs - 1]
        train_score = history.history['acc'][self.epochs - 1]
        print('...............')
        self.cnnlayers = list()
        for i in range(len(model.layers)):
            self.cnnlayers.append(model.layers[i].get_weights())
        Y_pred = model.predict(test_data.drop([self.label_name],axis=1).values.reshape(-1, 28, 28, 1) / 255.0)
        Y_pred_classes = model.predict_classes(test_data.drop([self.label_name], axis=1).values.reshape(-1, 28, 28, 1) / 255.0)
        Y_pred_labels = list(map(lambda x: self.train_y_label[x],Y_pred_classes.tolist()))
        # Y_pred_classes = np.argmax(Y_pred, axis=1)
        self.label_domain = list(set(train_data[self.label_name]))
        metas = [DiscreteVariable('predict', self.label_domain),
                 ContinuousVariable('negative probability'),
                 ContinuousVariable('positive probability')]
        new_two_prob_positive = []
        new_two_prob_negative = []

        wind_pre_prob = Y_pred.tolist()
        wind_pre = np.array(Y_pred_classes.ravel().tolist())

        for index, item in enumerate(wind_pre_prob):
            if len(self.label_domain) > 2:
                max_prob = item[int(wind_pre[index])]
                new_two_prob_negative.append(1 - max_prob)
                new_two_prob_positive.append(max_prob)
            elif len(self.label_domain) == 2:
                max_prob = item[1]
                new_two_prob_negative.append(1 - max_prob)
                new_two_prob_positive.append(max_prob)
        wind_pre_prob = np.array([np.array(new_two_prob_negative), np.array(new_two_prob_positive)])
        wind_pre_prob = wind_pre_prob.T

        cols = [np.array(Y_pred_classes).reshape(-1,1), wind_pre_prob]
        tbl = np.column_stack((np.array(Y_pred_classes).reshape(-1, 1), wind_pre_prob))
        res = Table.from_numpy(Domain(metas), tbl)
        final_result = self.merge_data(self.test_data, res)
        self.send("News", final_result)
        self.send("Metric Score", None)
        self.send("Metas", metas)
        self.send("Columns", cols)

        results = None
        N = len(test_data)  # note that self.test_data is orange Table
        results = Results(self.test_data, store_data=True)
        results.folds = None
        results.row_indices = np.arange(N)
        results.actual = np.array(test_data[self.label_name])
        results.predicted = np.array([Y_pred_classes])

        results.probabilities = np.array([wind_pre_prob])  # numpy.array([p.results[1] for p in slots])
        # results.probabilities = wind_pre_prob

        if len(self.label_domain) > 2:
            results.probabilities = None

        print("predicted", results.predicted[0])
        print("actual", results.actual)

        results.learner_names = ['sentiment_model']
        # self.send("Predictions", predictions)
        self.send("Evaluation Results", results)
        self.send("Metric", MetricType.ACCURACY)
        metric_frame = MetricFrame([[val_score], [train_score]], index=["hyperparam_searcher", "modeler"],
                                   columns=[MetricType.ACCURACY.name])
        self.send("Metric Score", metric_frame)

        # print(MetricType.ACCURACY)
        print(metric_frame)
        print(results.predicted, results.predicted.shape)

        # output = pd.DataFrame(Y_pred_classes, columns=['label'])
        # self.send('News', output)

    def merge_data(self,origin, result):
        # result table to origin table's meta
        domain = origin.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        new_domain = Domain(attributes, classes, meta_attrs+result.domain.variables)
        new_table = Table.from_table(new_domain, origin)
        for attr in result.domain:
            new_table.get_column_view(attr)[0][:] = result.get_column_view(attr)[0][:]
        return new_table

    def rerun(self):
        test_data = table2df(self.test_data)
        try:
            if len(self.test_data.domain.class_vars)>=1:
                test_data.iloc[:, :-len(self.test_data.domain.class_vars)] = \
                test_data.iloc[:, :-len(self.test_data.domain.class_vars)].astype('float')
            else:
                test_data = test_data.astype('float')


        except Exception as e:
            print(e)
            raise Error('data must be numeric')
        if self.label_name in test_data.columns:
            test_x = test_data.drop([self.label_name], axis=1)
            test_y = test_data[self.label_name]
        else:
            test_x = test_data
            test_x = test_data.astype('float')
        test_x = test_x.values.reshape(-1, 28, 28, 1) / 255.0
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y_label), activation="softmax"))

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                    min_lr=0.00001)
        self.epochs = 30  # Turn epochs to 30 to get 0.9967 accuracy
        batch_size = 86
        # Without data augmentation i obtained an accuracy of 0.98114
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(test_x)
        for i in range(len(model.layers)):
            model.layers[i].set_weights(self.cnnlayers[i])
        Y_pred = model.predict(test_x)
        Y_pred_classes = model.predict_classes(test_x)
        Y_pred_labels = list(map(lambda x: self.train_y_label[x], Y_pred_classes.tolist()))
        # Y_pred_classes = np.argmax(Y_pred, axis=1)
        self.label_domain = list(set(self.train_y_label))
        metas = [DiscreteVariable('predict', self.label_domain),
                 ContinuousVariable('negative probability'),
                 ContinuousVariable('positive probability')]
        new_two_prob_positive = []
        new_two_prob_negative = []

        wind_pre_prob = Y_pred.tolist()
        wind_pre = np.array(Y_pred_classes.ravel().tolist())

        for index, item in enumerate(wind_pre_prob):
            if len(self.label_domain) > 2:
                max_prob = item[int(wind_pre[index])]
                new_two_prob_negative.append(1 - max_prob)
                new_two_prob_positive.append(max_prob)
            elif len(self.label_domain) == 2:
                max_prob = item[1]
                new_two_prob_negative.append(1 - max_prob)
                new_two_prob_positive.append(max_prob)
        wind_pre_prob = np.array([np.array(new_two_prob_negative), np.array(new_two_prob_positive)])
        wind_pre_prob = wind_pre_prob.T

        cols = [np.array(Y_pred_classes).reshape(-1, 1), wind_pre_prob]
        tbl = np.column_stack((np.array(Y_pred_classes).reshape(-1, 1), wind_pre_prob))
        res = Table.from_numpy(Domain(metas), tbl)
        final_result = self.merge_data(self.test_data, res)
        self.send("News", final_result)
        self.send("Metric Score", None)
        self.send("Metas", metas)
        self.send("Columns", cols)
