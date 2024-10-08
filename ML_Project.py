import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

def read_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

data = read_data('text.txt')
print("Number of instances: {}".format(len(data)))

def ngram(token, n):
    output = []
    for i in range(n-1, len(token)):
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram)
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = []
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

def convert_label(item, name):
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)):
        if items[idx] == 1:
            label += name[idx] + " "
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=123)

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

# Vectorize the features
vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Apply Standard Scaler
scaler = StandardScaler(with_mean=False)  # with_mean=False because of sparse matrices
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
svc = SVC()
lsvc = LinearSVC(random_state=123, max_iter=20000, C=0.01, class_weight='balanced')
rforest = RandomForestClassifier(random_state=123, class_weight='balanced')
dtree = DecisionTreeClassifier()
nbayes = MultinomialNB()
ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=123 )

# clfs = [lsvc, rforest, svc, dtree]
clfs = [lsvc, svc, rforest, dtree, nbayes, ann]

# Train and test them, also find the best model based on test accuracy
best_clf = None
best_test_acc = 0
best_clf_name = ""

print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-" * 25, "-" * 17, "-" * 13))
for clf in clfs:
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))
    
    # Keep track of the best model based on test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_clf = clf
        best_clf_name = clf_name

print("\nBest model for prediction is: {}".format(best_clf_name))
print("Training Accuracy: {:.7f}".format(train_acc))
print("Test Accuracy: {:.7f}".format(test_acc))

# Create emoji dictionary
emoji_dict = {"joy": "😄", "fear": "😰", "anger": "😠", "sadness": "😢", "disgust": "🤢", "shame": "😔", "guilt": "😣"}

# User input prediction using the best model
Intext = input('Enter Text: ')
texts = [Intext]
for text in texts:
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    features = scaler.transform(features)  # Apply scaling to the new features
    prediction = best_clf.predict(features)[0]
    print(text,":", emoji_dict[prediction])
