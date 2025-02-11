import torch.nn as nn 
import torch 
import torchvision
from torchvision.transforms import transforms
from glob import glob 
from torchvision.transforms import Lambda, Compose, ToPILImage
from preprocessing.crop_transform_pad_images import read_image, crop_nonzero, pad_to_largest_square


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['NRG', 'RG']
    model = torchvision.models.resnet50(weights=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    checkpoint = torch.load('/home/shirshak/inference_BPEye_Project_2024/glaucoma_resnet_airogs_focal/epoch_018.ckpt')
    state_dict = {k.replace('net.model.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    my_transforms = transforms.Compose([
        # transforms.ToPILImage(), 
        transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize(size=(512,512),antialias=True)
    ])

    img_paths = sorted(glob('/mnt/Enterprise2/shirshak/Glaucoma_Dataset_Drishti-GS/Drishti-GS1_preprocessed_cropped_separated_train_test_val/test/*/*.png'))
    img_path = img_paths[0]

    for img_path in img_paths:
        img_transform = Compose([
        Lambda(read_image),
        Lambda(crop_nonzero),
        Lambda(pad_to_largest_square),
        ToPILImage(),
        ]) 


        preprocessed_img = img_transform(img_path)

        preprocessed_img.save("image.png")

        preprocessed_transformed_image = my_transforms(preprocessed_img)

        preprocessed_transformed_image = preprocessed_transformed_image.unsqueeze(0).to(device)
        # print(preprocessed_transformed_image.shape)

        outputs = nn.Sigmoid()(model(preprocessed_transformed_image))
        _, predicted_test = torch.max(outputs, 1)

        predicted_label = "Referable Glaucoma" if predicted_test.item() == 1 else "Non-Referable Glaucoma"


    

    result = {
        "image_path": img_path,
        "actual_label": labels_list[0].tolist(),
        "predicted_label": predicted_
    }


    print("Hello, model loaded")
