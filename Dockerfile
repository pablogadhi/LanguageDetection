FROM opennmt/ctranslate2:latest-ubuntu18-cuda10.2
WORKDIR /translation

COPY models models
COPY translation_server server

RUN pip3 install pyonmttok waitress Flask spacy
RUN python3 -m spacy download en
RUN python3 -m spacy download es
RUN python3 -m spacy download fr
RUN python3 -m spacy download de

EXPOSE 8080
ENTRYPOINT ["python3", "./server/server.py"]
CMD ["--beam_size", "4", "--batch_size", "50"]