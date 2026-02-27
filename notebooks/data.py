# function to compare models
def compare_classification_models(model1, model2,
                                  X_train, y_train,
                                  X_test, y_test,
                                  model1_name="Model 1",
                                  model2_name="Model 2",
                                  criterion="f1"):
    """
    Train, evaluate, and compare two classification models.
    """

    models = {
        model1_name: model1,
        model2_name: model2
    }

    results = {}

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities (for ROC-AUC)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None

        # Metrics
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc
        }

    results_df = pd.DataFrame(results).T

    # Select best model
    best_model_name = results_df[criterion.upper() if criterion != "roc_auc" else "ROC-AUC"].idxmax()
    best_model = models[best_model_name]

    print(f"\nBest model based on {criterion.upper()}: {best_model_name}")

    return best_model, results_df