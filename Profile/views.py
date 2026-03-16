from django.shortcuts import render
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier


def extract_features(username):

    length = len(username)
    numbers = len(re.findall(r'\d', username))
    underscores = username.count('_')
    special = len(re.findall(r'[^a-zA-Z0-9_]', username))
    letters = len(re.findall(r'[a-zA-Z]', username))

    digits_ratio = numbers/(length+1)
    letter_ratio = letters/(length+1)
    underscore_ratio = underscores/(length+1)

    return [length,numbers,underscores,special,letters,
            digits_ratio,letter_ratio,underscore_ratio]


def User(request):

    result = ""
    probability = ""

    if request.method == 'POST':

        username = request.POST.get('t1')

        # load dataset
        dataset = pd.read_csv('Profile/dataset/dataset.txt')

        X = dataset.iloc[:,0:8]
        y = dataset.iloc[:,8]

        # train model
        model = DecisionTreeClassifier()
        model.fit(X,y)

        # extract username features
        features = extract_features(username)

        # prediction
        prediction = model.predict([features])
        probability_score = model.predict_proba([features])

        fake_probability = probability_score[0][1] * 100

        if prediction[0] == 1:
            result = "Fake Profile Detected"
        else:
            result = "Real Profile"

        probability = round(fake_probability,2)

    return render(request,'User.html',
                  {'result':result,
                   'probability':probability})
