from bookrecs import process_data, modelling, make_recommendations

if __name__ == "__main__":
    dataset, train, test = process_data.read_data("data/goodbooks/ratings.csv")
    n_users, n_books = process_data.get_data_summary(dataset)
    if modelling.should_load():
        model = modelling.load_trained_model()
    else:
        model = modelling.create_model(n_books, n_users)
        model, history = modelling.train_model(model, train, n_epochs=5)
