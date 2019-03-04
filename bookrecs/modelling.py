import os
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model, load_model

MODEL_FILENAME = "data/regression_model.h5"

def create_model(n_books, n_users):
    """Create a model to train.

    Args:
        n_books (int): number of books
        n_users (int): number of users

    Returns: keras model 

    """
    book_input = Input(shape=[1], name="Book-Input")
    book_embedding = Embedding(n_books + 1, 5, name="Book-Embedding")(book_input)
    book_vec = Flatten(name="Flatten-Books")(book_embedding)

    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users + 1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])

    model = Model([user_input, book_input], prod)
    model.compile("adam", "mean_squared_error")

    return model


def train_model(model, train, n_epochs=5):
    """Train the model and save as output file.

    Args:
        model (keras model): model that has been set up
        train (dataframe): training dataset
        n_epochs (int): number of epochs to train for 

    Returns: trained model, history

    """
    history = model.fit(
        [train.user_id, train.book_id], train.rating, epochs=n_epochs, verbose=1
    )
    model.save(MODEL_FILENAME)

    return model, history


def load_model():
    """Load the trained model from disk
    Returns: model read from disk

    """
    return load_model(MODEL_FILENAME)


def should_load():
    """Determine if we should load from the disk
    Returns: boolean if we should load the model or not

    """
    return os.path.exists(MODEL_FILENAME)
