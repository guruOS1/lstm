FROM tensorflow/tensorflow

RUN pip3 install sklearn pandas

WORKDIR /app
COPY lstm.py /app/app.py
COPY data/reviews_dataset.tsv.zip /app/data/reviews_dataset.tsv.zip
COPY model/.keep /app/model/.keep
CMD ["bash"]