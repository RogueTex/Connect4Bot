FROM python:3.8
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN mkdir /FOLDERNAME/
RUN chmod -R 777 /FOLDERNAME
ADD connect4_policy_player.py /
ADD server.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "/server.py" ]
