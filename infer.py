import torch
import argparse
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.DeepLabV3Plus import model

def parse_args():
    parser = argparse.ArgumentParser(description='Polyp Segmentation Inference')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    return parser.parse_args()

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, v in color_dict.items():
        output[mask == k] = v
    return output

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    color_dict = {
    0: (0, 0, 0),  
    1: (255, 0, 0), 
    2: (0, 255, 0)  
    }
    
    try:
        checkpoint = torch.load('checkpoints/best.pth',
                              map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)
        
        ori_img = cv2.imread(args.image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = ori_img.shape[:2]
        
        img = cv2.resize(ori_img, (256, 256))
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_img = val_transform(Image.fromarray(img))
        input_img = input_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
        
        mask = cv2.resize(output_mask, (ori_w, ori_h))
        mask = np.argmax(mask, axis=2)
        mask_rgb = mask_to_rgb(mask, color_dict)

        cv2.imwrite('mask.jpeg', cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()