# Aero-Net
This repo introduces Aero-Net model architecture with convolutional layers integrated with transformer encoders to capture both local and global contextual information, Multi-scale attention module is used to process features of objects at multiple scales and feature fusion module concatenates multi-scale features and a cascading network for precise classification. We will use this framework to perform aircraft type recognition from remote sensing images which is a difficuilt task.

![Uploading Fighter-Net Architecture.png…]()

![slice_ethopia merged_x1942_y3189](https://github.com/user-attachments/assets/a91eb3bb-83c3-45df-a705-471da2e56c42)
![slice_hill base (7-2016)_x6066_y6025](https://github.com/user-attachments/assets/f2ffbb28-0f08-4207-9209-faadefe9dc83)
![labels](https://github.com/user-attachments/assets/f0145a5e-14c3-49d7-ab52-d37232c67343)
![labels_correlogram](https://github.com/user-attachments/assets/e201220a-011a-4ca7-9e5b-6cfa50b5a357)
![confusion_matrix_normalized](https://github.com/user-attachments/assets/08fe54ba-4895-4ac2-82eb-bbd3d4cd6f71)
![Uploading val_batch1_labels.jpg…]()
![train_batch1980](https://github.com/user-attachments/assets/9c50bc41-d5fb-4899-a7fe-be82ce0ffeb5)
![results](https://github.com/user-attachments/assets/da8202df-633b-4ca2-9059-bb7741da5cc2)


## Step 1: Create a new environment
`conda create -n fighter-net-env python=3.8`

## Step 2: Activate the environment
`conda activate fighter-net-env`

## Step 3: Install dependencies
`pip install -r requirements.txt`

## Step 4: Check if CUDA is available (Optional for GPU setup)
`python -c "import torch; print(torch.cuda.is_available())"`

## Step 5: Start training the model
`python train.py --config config.yaml --epochs 50 --batch-size 32 --lr 0.001 --model-size medium --save-dir './trained_models'`


Explanation:
    --config config.yaml: Path to the configuration file (in this case, config.yaml).
    --epochs 50: Number of epochs for training. You can adjust this value as needed.
    --batch-size 32: The batch size used for training.
    --lr 0.001: The learning rate for the optimizer.
    --save-dir './trained_models': Directory where model checkpoints will be saved after each epoch.
    
    
## Step 6: Run predictions after training
`python predict.py --config config.yaml --model-path ./trained_models/fighter_net_epoch_50.pth --image-path ./data/test/sample_image.jpg --model-size medium --output-dir ./predictions`


Explanation:
    --config config.yaml: Path to the configuration file (the same one used during training).
    --image /path/to/image.jpg: Path to the input image you want to predict on.
    --model-path ./trained_models/model_epoch_last.pth: Path to the trained model checkpoint.
    --batch-size 1: Set the batch size for prediction (typically 1 for inference).
    --output-dir ./output_predictions: Directory where the predictions will be saved.
    
## Step 7: Save the trained model (inside train.py)
`torch.save(model.state_dict(), 'path_to_save_model.pt')`

## Step 8: Deactivate the environment
`conda deactivate`


