FROM opennmt/ctranslate2:latest-ubuntu18-cuda10.2
WORKDIR /translation

COPY models models
COPY translation_server server

RUN pip3 install pyonmttok waitress Flask

EXPOSE 8080
ENTRYPOINT ["python3", "./server/server.py"]
CMD ["--beam_size", "4", "--batch_size", "50"]