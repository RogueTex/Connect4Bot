# Use TensorFlow base image so we don't pip-install 207MB during every build.
FROM tensorflow/tensorflow:2.18.0
ADD requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
RUN mkdir /FOLDERNAME/
RUN chmod -R 777 /FOLDERNAME
ADD connect4_policy_player.py /
ADD server.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "/server.py" ]
