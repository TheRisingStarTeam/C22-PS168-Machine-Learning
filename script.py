import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import requests
from typing import Dict, Text
import schedule
import time


def main():
    print('Grab data....')
    # get from endpoint
    dfEvent = pd.read_json(
        'https://asia-southeast2-the-rising-stars.cloudfunctions.net/app-1/events')
    dfUser = pd.read_json(
        'https://asia-southeast2-the-rising-stars.cloudfunctions.net/app-1/userIdentities')

    print('Preprocesing data....')
    # get only needed data
    dfUser.dropna(subset=['interest', 'history'], inplace=True)
    dfEvent.dropna(subset=['eventId', 'categories'], inplace=True)

    print(f'banyak user => {dfUser.shape[0]}')
    print(f'banyak event => {dfEvent.shape[0]}')

    # make relation 1
    history = dfUser.loc[:, ['userId', 'history']]
    history.columns = ['userId', 'eventId']
    history = history.explode('eventId')
    relationUidEid1 = history.values.tolist()

    # make relation 2

    userId = np.array(dfUser.loc[:, 'userId'])
    userInterest = np.array(dfUser.loc[:, 'interest'])
    eventCategory = np.array(dfEvent.loc[:, 'categories'])
    eventId = np.array(dfEvent.loc[:, 'eventId'])

    relationUidEid2 = []
    for interest, UId in zip(userInterest, userId):
        for category, EId in zip(eventCategory, eventId):
            if (np.intersect1d(interest, category).shape[0] < 1):
                continue
            relationUidEid2.append([UId, EId])

    # combine relation
    combinedRelation = relationUidEid1+relationUidEid2

    print(f'banyak telasi => {len(combinedRelation)}')
    # create tf dataset
    combinedRelation = tf.data.Dataset.from_tensor_slices(combinedRelation)
    eventId = tf.data.Dataset.from_tensor_slices(eventId)
    combinedRelation = combinedRelation.map(
        lambda x: {'userId': x[0], 'eventId': x[1]})

    # hyper parameter tuning
    embedding_dimension = 512
    unique_user_ids = dfUser.userId.to_list()
    unique_event_id = dfEvent.eventId.to_list()
    lenData = combinedRelation.__len__().numpy()
    trainSize = int(lenData*80/100)
    testSize = int(lenData-trainSize)
    batch = 64
    epoch = 50
    jumlahRekomendasi = 10

    # build model

    class Model(tfrs.Model):

        def __init__(self):
            super().__init__()

            # Set up user representation.
            self.user_model = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                # We add an additional embedding to account for unknown tokens.
                tf.keras.layers.Embedding(
                    len(unique_user_ids) + 1, embedding_dimension)
            ])
            # Set up movie representation.
            self.event_model = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_event_id, mask_token=None),
                tf.keras.layers.Embedding(
                    len(unique_event_id) + 1, embedding_dimension)
            ])
            # Set up a retrieval task and evaluation metrics over the
            # entire dataset of candidates.
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=eventId.batch(batch).map(self.event_model)
                )
            )

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

            user_embeddings = self.user_model(features["userId"])
            movie_embeddings = self.event_model(features["eventId"])

            return self.task(user_embeddings, movie_embeddings)

    model = Model()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    tf.random.set_seed(42)
    shuffled = combinedRelation.shuffle(
        lenData, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(trainSize)
    test = shuffled.skip(trainSize).take(testSize)

    # Train.
    print('Training Model....')
    model.fit(train.batch(batch), epochs=epoch, verbose=1)

    # Evaluate.
    model.evaluate(test.batch(batch), return_dict=True)

    # predict
    print('Uploading.....')

    def predict(userId, numOut):
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            eventId.batch(batch).map(lambda event: (
                event, model.event_model(event)))
        )

        # Get recommendations.
        _, events = index(tf.constant([userId]))
        return events[0, :numOut]

    # upload recomendation
    for userid in unique_user_ids:
        arrPred = predict(userid, jumlahRekomendasi)
        decoder = np.vectorize(lambda x: x.decode('UTF-8'))
        decodedPred = decoder(arrPred.numpy()).tolist()

        r = requests.put('https://asia-southeast2-the-rising-stars.cloudfunctions.net/app-1/recommendation/' +
                         userid, data={'recommendations': decodedPred})

    print('done ....')

    print('\n\n please wait until next schedule')

    def info():
        print('please wait until next schedule')


if __name__ == '__main__':
    # schedule.every(1).minutes.do(main)                 # pake ini kalau permenit
    schedule.every().day.at("12:00").do(main)            # pake ini klo perhari
    print('\n\n please wait until next schedule')

    while True:
        schedule.run_pending()
        time.sleep(1)
