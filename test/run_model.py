import torch

# Model
# model = torch.hub.load('C:/Users/koen_/OneDrive/Bureaublad/klieder_folder', 'best.pt')
model = torch.hub.load('C:/Users/koen_/GitHub/yolov5', 'custom', path='C:/Users/koen_/OneDrive/Documents/.Koen de Raad/- Whitebox Data Science/Projecten/TeamTV/Trained Models/Hockey/detectors/exp_24-05/best.pt', source='local')  # local repo
# Image
im1 = 'https://uploads-ssl.webflow.com/613214e333d406adbe3c4fa3/624d4151097ff22b841869a3_teamtv-automatic-camera-hockey-1.jpg'
im2 = 'https://uploads-ssl.webflow.com/613214e333d406adbe3c4fa3/624d4151e08ab2bb95f3711f_teamtv-automatic-camera-hockey-2.jpg'
im3 = 'https://uploads-ssl.webflow.com/613214e333d406adbe3c4fa3/624d4150b7135549bf532330_teamtv-automatic-camera-hockey-3.jpg'
im4 = 'https://uploads-ssl.webflow.com/613214e333d406adbe3c4fa3/624d4150bb6b4c44f79d6714_teamtv-automatic-camera-hockey-4.jpg'

# Inference
results = model([im1, im2, im3, im4])
results.print()
print(results.pandas().xyxy)
results.save()
