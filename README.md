# PretrainedMobilenetV2-Unet-Flaskapp
Implementation of Unet on flask app with pretrained MobilenetV2
The tutorial of detailed implementation of Unet with pretrained MobilenetV2 can be found at https://idiotdeveloper.com/unet-segmentation-with-pretrained-mobilenetv2-as-encoder/.
Code has beed modified to read pics from online resources. The links for pics should be in an excel file. 
Once the model is trained, it will be saved for Flask App.
In folder 'src', run client.ipynb, so the Flask app will use the saved model to predict boundaries like the CVC-ClinicDB data (https://polyp.grand-challenge.org/CVCClinicDB/).
The original paper of Unet can be found at https://arxiv.org/abs/1505.04597, MobileNetV2 https://arxiv.org/abs/1801.04381.
