# contamination='auto', deliberately NOT a fixed 0.05 -- fixed
    # earlier in the project. We're training on data we've validated as
    # genuinely normal, so asserting "5% of this is anomalous" would be
    # baking a false premise into the decision boundary. 'auto' uses
    # the original Isolation Forest paper's own offset heuristic
    # instead. Real severity thresholds (low/medium/high/critical) get
    # calibrated separately in the next phase, against the LABELED data
    # where we actually know the true anomaly rate -- not smuggled in
    # here as an assumption about clean training data.
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more
    # anomalous. Saving this training distribution alongside the model
    # gives detect.py (and whoever builds the eval script next) a
    # reference point for "what did normal actually look like," instead
    # of calibrating severity thresholds blind.
    scores = model.decision_function(X)
    print("Training score distribution (decision_function; LOWER = more anomalous):")
    print(pd.Series(scores).describe())