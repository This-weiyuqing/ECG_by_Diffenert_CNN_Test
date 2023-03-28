def Create_CNN_By_Yourself():
    filename = input() + ".py"
    print("file name", filename)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            "import datetime\nimport os\nimport numpy as np\nimport tensorflow as tf\nfrom utils_seventyOne_of_Filters \nimport loadData, plot_history_tf, plot_heat_map\n")
        f.write('Project_PATH = "../')

        print("\n\n\n" + "Enter the path to save data from the start of the secondary folder,Used / arnt\\")
        Project_PATH = input()
        f.write(Project_PATH + "\\" + '"\n')
        print('Project_PATH=' + Project_PATH)

        f.write(
            'log_dir = Project_PATH + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")\nmodel_Path_one = Project_PATH  + "model/\n')
        print("\n\n\n" + "Enter model name ")
        model_name = input()
        f.write(model_name + '.h5"\n')
        print('model_name=' + model_name)

        print("\n\n\n" + 'Enter RATIO (usually used 0.3)')
        RATIO = input()
        f.write("RETIO=" + RATIO + '\n')
        print('RATIO=' + RATIO)

        print("\n\n\n" + 'Enter RANDOM_SEED ')
        RANDOM_SEED = input()
        f.write('RANDOM_SEED=' + RANDOM_SEED + '\n')
        print('RANDOM_SEED=' + RANDOM_SEED)

        print('\n\n\n' + "Enter BATCH_SIZE,this number of samples per gradient update")
        BATCH_SIZE = input('')
        f.write("BATCH_SIZE=" + BATCH_SIZE + '\n')
        print('BATHC_SIZE=' + BATCH_SIZE)

        print('\n\n\n' + "Enter NUM_POCHS")
        NUM_POCHS = input()
        f.write('NUM_EPOCHS =' + NUM_POCHS + '\n')

        # def CNN mode
        print('Name of this CNN model')
        CNN_NAME = input()

        #Input array type
        f.write(
            'def ' + CNN_NAME + '():\n    leavlOneModel = tf.keras.models.Sequential([\n        tf.keras.layers.InputLayer(input_shape=(300,)),\n        tf.keras.layers.Reshape(target_shape=(300, 1)),\n')

        # def CNN model
        print('\n\n\nEnter your CNN model layer.')
        CNN_Layer = input()
        CNN_Layer_Int = int(CNN_Layer)
        j = 1
        while (j <=CNN_Layer_Int):
            print('\n\n\nEnter the parameters for the ' + j + ' layer of your CNN model.')
            # this is Conv1D code, you can change by your self
            print('Enter your Conv1D filters , kernel_size,strides,padding,sctivation. And Enter once for each parameter')
            filters = int(input())
            kernel_size = int(input())
            strides = int(input())
            padding = input()
            activation = input()
            f.write(
                "       tf.keras.layers.Conv1D(filters=" + filters + ", kernel_size=" + kernel_size + ", strides=" + strides + ", padding='" + padding + "', activation='" + activation + "),\n")

            # Pool paremeters
            print('\n\n\nNow,you shoud enter your Pool Paremeters')

            # this is pool1D,you can change to your self

            ##because of my CNN used only MaxPool1D,I commented out this line of code,you can uncommented and use this code to change the type for your PoolFunction
            # print('\n\n\nEnter your Pool types,such as MaxPool ')
            # Pool_types=input()
            # f.write('tf.keras.layers.'+Pool_types+'1D')

            # If you use the above paragraph code, you need commented this paragraph
            f.write('       tf.keras.layers.MaxPool1D')

            print('\n\n\nEnter the filters strides padding by Enter once for each parameter')
            pool_size = int(input())
            strides_Pool = int(input())
            padding_Pool = int(input())
            f.write("(pool_size=" + pool_size + ", strides=" + strides_Pool + ", padding='" + padding_Pool + "'),\n")
            j = j + 1
            # Now we create the fully connected layer
        # def Dense
        print('\n\n\nEnter your Dense layer')
        k = 1
        Dense_Layer = int(input())
        while (k <= Dense_Layer):
            print('\n\n\nEnter the type of calculation layer,such as Flatten Dropout')
            calculation = input()
            print('In this calculation, you needed parameters. Such as Dropout, you can enter 0.3 or otherthing,but you should add the complete parameter setup (including instructions).Like   rate=0.2  ,this is Dropout parameter needs from')
            calculation_parameters = input()
            f.write("       tf.keras.layers."+calculation+"("+calculation_parameters+"),")
            #Now,lets make the Dence
            print('Now.;lets take the Dence'
                  'you need take how much of your categories and whitch calculation you need.such as Relu,softmax,tanH')
            categories_Number=input()
            activation_Dense=input()
            f.write("       tf.keras.layers.Dense("+categories_Number+", activation='"+activation_Dense+"'),\n" )
            k=k+1

        f.write("    ])\n    return leavlOneModel\n")
        # def main
        f.write(
            "def main():\n    X_train, X_test, y_train, y_test = loadData(RATIO, RANDOM_SEED)\n    if os.path.exists(model_Path_one):\n        print(' get model in h5 file')\n        model = tf.keras.models.load_model(filepath=model_Path_one)\n    else:\n        # create new model(if model unexists)\n        model = CNN_model_level_one()\n        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])\n        model.summary()\n        # TB make\n        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,\n                                                              histogram_freq=1,\n                                                              )\n        # Training and validation\n        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,\n                            batch_size=BATCH_SIZE,\n                            validation_data=(X_test, y_test),\n                            callbacks=[tensorboard_callback])\n        # validation_split=RATIO,\n        model.save(filepath=model_Path_one)\n        plot_history_tf(history)\n    y_pred = np.argmax(model.predict(X_test), axis=-1)\n    plot_heat_map(y_test, y_pred)")

        # print(str=f.read(filename+".py"))

        f.write("if __name__ == '__main__':\n    main()")

if __name__ == '__main__':
    Create_CNN_By_Yourself()