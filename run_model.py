import os
import scipy
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

labels = ["Clear_Water","Far_From_Shore_Major_Bloom","Far_From_Shore_Medium_Bloom","Far_From_Shore_Minor_Bloom","Land","Near_Shore_Narrow_Major_Bloom","Near_Shore_Narrow_Medium_Bloom","Near_Shore_Narrow_Minor_Bloom","Near_Shore_Tall_and_Wide_Major_Bloom","Near_Shore_Tall_and_Wide_Medium_Bloom","Near_Shore_Tall_and_Wide_Minor_Bloom"]
maxWidth = 0
maxHeight = 0
files = []
gridSize = 32

# ----------------------------------------------------------------------
# STEP ONE
# GRID IMAGE
# ----------------------------------------------------------------------

def long_slice(image_path, out_name, outdir, sliceHeight, sliceWidth):
    global maxWidth
    global maxHeight
    img = Image.open(image_path) # Load image
    imageWidth, imageHeight = img.size # Get image dimensions
    left = 0 # Set the left-most edge
    upper = 0 # Set the top-most edge
    while (left < imageWidth):
        while (upper < imageHeight):
            # If the bottom and right of the cropping box overruns the image.
            if (upper + sliceHeight > imageHeight and \
                left + sliceWidth > imageWidth):
                bbox = (left, upper, imageWidth, imageHeight)
            # If the right of the cropping box overruns the image
            elif (left + sliceWidth > imageWidth):
                bbox = (left, upper, imageWidth, upper + sliceHeight)
            # If the bottom of the cropping box overruns the image
            elif (upper + sliceHeight > imageHeight):
                bbox = (left, upper, left + sliceWidth, imageHeight)
            # If the entire cropping box is inside the image,
            # proceed normally.
            else:
                bbox = (left, upper, left + sliceWidth, upper + sliceHeight)
            working_slice = img.crop(bbox) # Crop image based on created bounds
            # Save your new cropped image.
            files.append(str('slice_' + out_name + '_' + str(int(upper/sliceHeight)) + '_' + str(int(left/sliceWidth)) + '.png'))
            working_slice.save(os.path.join(outdir, 'slice_' + out_name + \
                '_' + str(int(upper/sliceHeight)) + '_' + str(int(left/sliceWidth)) + '.png'))
            upper += sliceHeight # Increment the horizontal position
            maxHeight = max(maxHeight, int(upper/sliceHeight))
        left += sliceWidth # Increment the vertical position
        maxWidth = max(maxWidth, int(left/sliceWidth))
        upper = 0
    print("Done")

print("Gridding image...")
file = "input.png"
long_slice('' + file, 'map', "temp_imgs", gridSize, gridSize)


# ----------------------------------------------------------------------
# STEP TWO
# IDENTIFY IMAGES
# ----------------------------------------------------------------------
print("Identifing")
classArr = [0] * (maxWidth-1)
for i in range(maxWidth-1):
    classArr[i] = [0] * (maxHeight-1)

    
# load json and create model
json_file = open('data/model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("data/model_weights.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def identifyImg(file):
    img = load_img(file)
    x = np.expand_dims(scipy.misc.imresize(img_to_array(img),[64,64]),axis=0)
    predResults = loaded_model.predict(x,steps=1)
    return labels[predResults.argmax(axis=-1)[0]]


for i in range(maxHeight-1):
    for j in range(maxWidth-1):
        classArr[j][i] = identifyImg("temp_imgs/slice_map_" + str(i)+ "_" + str(j) +".png")
print("Done")

# ----------------------------------------------------------------------
# STEP THREE
# PREDICT
# ----------------------------------------------------------------------
print("Starting predictions")
def predict(old,ty):
    if old == "Near_Shore_Narrow_Minor_Bloom" :
        return "Near_Shore_Narrow_Medium_Bloom"
    if old == "Near_Shore_Narrow_Medium_Bloom" :
        return "Near_Shore_Narrow_Major_Bloom"
    if old == "Near_Shore_Narrow_Major_Bloom" :
        return "Near_Shore_Tall_and_Wide_Medium_Bloom"

    if old == "Near_Shore_Tall_and_Wide_Medium_Bloom" :
        return "Near_Shore_Tall_and_Wide_Major_Bloom"
    if old == "Near_Shore_Tall_and_Wide_Minor_Bloom" :
        return "Near_Shore_Tall_and_Wide_Medium_Bloom"

    if old == "Far_From_Shore_Minor_Bloom" :
        return "Far_From_Shore_Medium_Bloom"
    if old == "Far_From_Shore_Medium_Bloom" :
        return "Far_From_Shore_Major_Bloom"
    return old

for i in range(maxHeight-1):
    for j in range(maxWidth-1):
        pred = predict(classArr[j][i],0)
        if pred != classArr[j][i]:
            os.remove("temp_imgs/slice_map_" + str(i)+ "_" + str(j) +".png")
            replacement_img = Image.open("replacements/" + str(gridSize) + "x" + str(gridSize) + "/" + pred + ".png")
            replacement_img.save("temp_imgs/slice_map_" + str(i)+ "_" + str(j) +".png")

print("Done")

# ----------------------------------------------------------------------
# STEP FOUR
# OUTPUT PREDICTION MAP
# ----------------------------------------------------------------------

result = Image.new("RGB", ((maxWidth-1)*gridSize, (maxHeight-1)*gridSize))

for w in range(maxHeight-1):
    for h in range(maxWidth-1):
        img = Image.open("temp_imgs/slice_map_" + str(w)+ "_" + str(h) +".png")
        result.paste(img, (h*gridSize, w*gridSize))

result.save('output.png')
print("Output saved!")

print("Recovering disk space...")
list( map( os.unlink, (os.path.join("temp_imgs",f) for f in os.listdir("temp_imgs")) ) )

print("Done")
