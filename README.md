# Individual Corallite Reconstruction #

## Data Organisation ##
An example file structure is provided in ```data -> example species.``` Much of the code relies on this file structure, 
so please note that adjustments are OK, but will require code changes... 
- ```complete_raw```: Should hold the complete set of colony scans, to be used for the final inference stage. 
- ```predictions```: Will be the directory where the final inference output is saved. 
- ```training_data```: This is where the model training data should be kept.
  - ```annotations```: This directory should hold the manual 'gold standard' annotations to train the model against.
  Annotations should be gray-scale: {0, 255}, where each pixel assigned a value of 255 has been identified 
  as belonging to a corallite region.  
  - ```raw_images```: Should hold the corresponding raw images for the annotations. 
  **NOTE:** For each annotated slice used for training, there should be the corresponding raw image, 
  plus *n* adjacent raw images either side to form a volumetric snippet. 
  Currently, snippet depth is set to 5, meaning *n=2*, however this can be amended as desired. 
  Noting that larger snippets in any dimension will increase training time.
  - ```snippets```: This the directory the constructed snippets will be saved to. 
  - ```snippet_checks```: This directory is used to output images of the snippets for visual verification that the 
  snippets are constructed as anticipated.
  - ```data_lists```: Used during training to read the file paths of the data, according to test, train and validate splits.

## Deep-Learning Pipeline ##

### Add Model Checkpoints ### 
Due to file sizes, model checkpoints must be downloaded from TODO: here. They should be saved to the appropriate 
folders in ```./VT-UNet/model/vit_checkpoint```.

### Using The Existing Model ###
If you just want to run inference on some coral data using the fine-tuned model:
- run ```predict_coral_volume.py --root {PATH_TO_SPECIES_ROOT}```

### Training A New Model ###
**NOTE:** It is best to train the model on a GPU cluster (ie. Blue Crystal). 
It will take a *very* long time on almost all personal computers. 
- First make sure to create the snippets for training/testing by running ```create_snippets.py --root {PATH_TO_SPECIES_ROOT}```
- Then run ```./VT-UNet/TransUNet/train.py``` making sure to either pass the appropriate paths as arguments, 
or updating the Coral config attributes (lines 74-80). 
- The default training setup is to fine-tune the Baseline model with 2+ *fully* annotated slices from the testing domain.
- If you are interested in creating a new baseline model, I suggest the below recommendations: 
  - Use the imagenet21k checkpoint: ```ViT-B_16.npz```
  - Make sure that all layers are unfrozen (see notes below)
  - Use a large volume of training data, which captures the core relationships you want to model. 
  Weak annotation is less of an issue at this stage, but it should be closely aligned with the end task.

#### Training Notes ####
- You can set/unset layer freezing during training by assigning: ```config_vit['freeze'] = [Layer1, Layer2]``` 
(Example line 128 of ```train.py```)

## 3D Reconstruction ##
Some things to bear in mind for this part of the pipeline:
- The development of this phase has not been optimised *at all* and can be quite slow to process, 
especially for large volumes of data.
- You will need to load and run ```blender_reconstruction.py``` from inside Blender as the external API doesn't 
seem to be working.
  - When the script is processing, there is little to suggest that anything is happening, 
  but it is: you just need to leave it running. 
  - I suggest generating a small model first to convince yourself everything is working as expected. 

### Generating the Corallite Map ###
- Run ```convert_to_blender_data.py --root {PATH} --fn {SAVE_NAME}``` where the root argument is the path the 
species directory, and fn is the name of the file you want to save to.
Assuming the directory structure described above is in place, this will generate a dictionary of unique corallites 
according the parameterised values in the script.
- Parameters can be adjusted by passing the following arguments when running the script:
  - ```--scale {FLOAT}```: Factor to scale the regions within blender. Default (0.01) 
  - ```--nn {FLOAT}```: Maximum Euclidean distance threshold to consider a prediction a viable nearest neighbour. 
  Default (0.15)
  - ```--iou {FLOAT}```: Minimum IoU to consider region overlap as possible neighbour. Default 0.3.
  - ```--search {INT}```: Number of sequential scan layers to search for a matching region. Default 3.

### Building the Model ###
- Open a new project in Blender and open up the 'Scripting' panel.
- Load the ```blender_reconstruction.py``` script and update the paths in main to point to the correct data. 
- Some notes:
  - ```reset_scene()``` will clear everything in the model space.
  - It is *not* fast, and doesn't give much feedback, in terms of progress. 
  I recommend starting with a small model / subset of corallites until you are familiar with it. 
  - You can load corallites in batches by ID, by changing the range values on line 139. I found that generating 
  the model in batches of 100-200 worked well for me. 
    - If you do this, make sure to comment out ```reset_scene()``` for subsequent loads so as not to lose the 
    previously generated models! 