FROM --platform=linux/amd64 pytorch/pytorch 

ENV PYTHONBUFFER 1
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

USER user 

WORKDIR /opt/app

COPY --chown=user:user ./requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

    
COPY --chown=user:user ./glaucoma_resnet_airogs_focal/* /opt/app/resources/

COPY --chown=user:user ./preprocessing/crop_transform_pad_images.py /opt/app/preprocessing/
COPY --chown=user:user ./inference_docker.py /opt/app/

# COPY NEEDED FILES FOR LOADING 1ST MODEL 
COPY --chown=user:user ./src/models/res_net_module.py /opt/app/src/models/

# COPY --chown=user:user  /mnt/Enterprise2/shirshak/Glaucoma_Dataset_Drishti-GS/Drishti-GS1_preprocessed_cropped_separated_train_test_val/test/*/*.png /opt/app/images

ENTRYPOINT ["python", "inference_docker.py"]