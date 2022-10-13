import torch
import torch.nn as nn
import dataset_tool as dt
import torch.utils.data as Data
import three_d_unet 

BATCH_SIZE = 1
image_size = 128

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainingDataset = dt.PBVTrainingDataset()
    validationDataset = dt.PBVTrainingDataset()

    

    train_loader = Data.DataLoader(
        dataset = trainingDataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    for batch in train_loader:
        training = batch['training']
        training_mark = batch['training_mark']

        training = training.to(device=device, dtype=torch.float32)
        #mask_type = torch.float32 if net.n_classes == 1 else torch.long
        training_mark = training_mark.to(device=device, dtype=torch.float32)

        #print(training)
    
        model = three_d_unet.UNet_3D(in_dim=1, out_dim=1, num_filters=6)
        model.to(device=device)
        
        out = model(training)
        
        

