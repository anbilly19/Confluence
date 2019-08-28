def calclaim (cost,dmg,loc,sev):
  if(dmg==0 and sev==0):
    return 0.25
  cnt = 1
  if(loc == 1 ):
    cnt *= 2
  else:
    cnt *= 3

  if (sev == 0):
    cnt *= 0.5
  elif(sev==1):
    cnt *= 3
  else:
    cnt *= 7

  if (cost==0):
    cnt *= 1
  elif(cost==1):
    cnt *= 1.5
  else:
    cnt *= 4
  
  return cnt

def loadIsDamaged():
  image_size = 150
  #Load the VGG model
  vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

  # Freeze the layers except the last 4 layers
  for layer in vgg_conv.layers[:-4]:
      layer.trainable = False

  # Create the model
  model = models.Sequential()
  
  # Add the vgg convolutional base model
  model.add(vgg_conv)
  
  # Add new layers
  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(2, activation='softmax'))
  
  # Show a summary of the model. Check the number of trainable parameters
  model.summary()

  model.load_weights('/content/drive/My Drive/dataset_zapsure/dmgornot_weights.h5')
  return model

def isDamaged(imgpath,model):
  image_size = 150 
  img = Image.open(imgpath).resize((image_size,image_size))
  img_arr = np.expand_dims(img_to_array(img), axis=0)

  image = preprocess_input(img_arr)
  prediction = model.predict(image)

  maxval = prediction.max()
  if(maxval == prediction[0][0]):
    return 1
  else:
    return 0
