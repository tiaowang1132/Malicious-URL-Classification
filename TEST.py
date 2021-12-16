import pandas as pd
from matplotlib import pyplot as plt
#from plotnine import *
import seaborn as sns
import sys
from urllib.parse import urlparse,urlencode
import ipaddress
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings


def get_data(text1,text2):
    df_black =  pd.read_csv(text1, error_bad_lines=False, engine='python' )
    df_white =  pd.read_csv(text2, error_bad_lines=False, engine='python')

    # print(type(df_black))
    # print(df_black.shape)
    # print(df_white.shape)

    df_black.columns=['url']
    df_white.columns=['url']

    # print(df_black)
    # print(df_white)

    df_white.describe()
    df_black.describe()
    df_white = df_white.dropna()
    df_black = df_black.dropna()
    df_white['lable'] = df_white.apply(lambda x:0, axis=1)
    df_black['lable'] = df_black.apply(lambda x:1, axis=1)

    df_white=df_white.drop_duplicates()
    df_black=df_black.drop_duplicates()

    df=pd.concat([df_white,df_black])
    df=df.drop_duplicates()
    return df

def getDomain(url):
    url=url.replace("[","").replace("]","")
    url=url.replace("\'","").replace("\"","")
    if not re.match('(?:http|ftp|tftp|https)://', url):
        url= 'http://{}'.format(url)
    #print(url)
    #domain = urlparse(url)
    #print(domain)

    path = urlparse(url).path
    domain=urlparse(url).netloc
    #print(domain)
    if re.match(r"^www.",domain):
        domain = domain.replace("www.","")
    return domain,url,path


# depth of domain (number of dots)
def domaindept(domain):
    depth = len(domain.split('.')) - 1
    # print(depth)
    #     if depth > 5:
    #         depth = 5
    return depth


# the number of dash in host
def domaindesh(domain):
    desh = len(domain.split('-')) - 1
    # print(desh)
    #     if desh > 4:
    #         desh = 4
    return desh


def pathDepth(path):
    s = path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth = depth + 1
    #     if depth > 8:
    #         depth = 8

    return depth


# length of domain
def domainlen(domain):
    if len(domain) > 22:
        domainlen = 1
    else:
        domainlen = 0
        # return domainlen
    return len(domain)


def containIP(url):
    #     try:
    #         ipaddress.ip_address(url)
    res = re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", url)
    if res:
        ip = 1
    else:
        ip = 0
    return ip


def containAt(url):
    if "@" in url:
        at = 1
    else:
        at = 0
    return at


def geturlLen(url):
    if len(url) < 54:
        length = 0
    else:
        length = 1
        # return length
    return len(url)


def containWords(url):
    words = r"confirm|banking|secure|ebayisapi|webscr|eBay|PayPal|sulake|facebook|orkut|santander|mastercard|warcraft|visa|bradesco"

    # path = urlparse(url).path
    match = re.search(words, url)
    if match:
        return 1
    else:
        return 0


# 6. Checking for Shortening Services in URL (Tiny_URL)
def tinyURL(url):
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                          r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                          r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                          r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                          r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                          r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                          r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                          r"tr\.im|link\.zip\.net"
    match = re.search(shortening_services, url)
    if match:
        return 1
    else:
        return 0

#Function to extract features
def featureExt(url,label):

    features = []
    domain,url,path = getDomain(url)

    features.append(domaindept(url))
    features.append(domaindesh(url))
    features.append(pathDepth(path))
    features.append(domainlen(url))
    features.append(containIP(url))
    features.append(containAt(url))
    features.append(geturlLen(url))
    features.append(containWords(url))
    features.append(tinyURL(url))
    features.append(label)

    return features

def concatdf():
    rfeatures = []

    for _, row in df.iterrows():
        # print(row)
        url = row['url']
        label = row['lable']
        rfeatures.append(featureExt(url, label))

    feature_names = ['domaindept', 'domaindesh', 'pathDepth', 'domainlen', 'containIP', 'containAt',
                     'geturlLen', 'containWords', 'tinyURL', 'Label']

    newdata = pd.DataFrame(rfeatures, columns=feature_names)
    newdata = newdata.sample(frac=1).reset_index(drop=True)
    print(newdata.head())
    return newdata

def DT(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_depth = 5)
    tree.fit(X_train, y_train)
    y_test_tree = tree.predict(X_test)
    y_train_tree = tree.predict(X_train)
    acc_train_tree = accuracy_score(y_train, y_train_tree)
    acc_test_tree = accuracy_score(y_test, y_test_tree)

    print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
    print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))
    FPR, recall, thresholds = roc_curve(y_test, y_test_tree, pos_label=1)
    area = AUC(y_test, y_test_tree)
    plt.figure()
    plt.plot(FPR, recall, color='red'
             , label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Decision Tree ROC')
    plt.legend(loc="lower right")
    plt.show()

def RR(X_train, X_test, y_train, y_test):
    forest = RandomForestClassifier(max_depth=5)
    # fit the model
    forest.fit(X_train, y_train)
    y_test_forest = forest.predict(X_test)
    y_train_forest = forest.predict(X_train)
    acc_train_forest = accuracy_score(y_train, y_train_forest)
    acc_test_forest = accuracy_score(y_test, y_test_forest)

    print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
    print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

    FPR, recall, thresholds = roc_curve(y_test, y_test_forest, pos_label=1)
    area = AUC(y_test, y_test_forest)
    plt.figure()
    plt.plot(FPR, recall, color='red'
             , label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Random forest ROC')
    plt.legend(loc="lower right")
    plt.show()

def KNN(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_test_knn = clf.predict(X_test)
    y_train_knn = clf.predict(X_train)
    acc_train_knn = accuracy_score(y_train,y_train_knn)
    acc_test_knn = accuracy_score(y_test,y_test_knn)


    print("KNN: Accuracy on training Data: {:.3f}".format(acc_train_knn))
    print("KNN : Accuracy on test Data: {:.3f}".format(acc_test_knn))
    FPR, recall, thresholds = roc_curve(y_test, y_test_knn, pos_label=1)
    area = AUC(y_test, y_test_knn)
    plt.figure()
    plt.plot(FPR, recall, color='red'
             , label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('KNN ROC')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    text1= sys.argv[1]
    text2= sys.argv[2]
    df=get_data(text1,text2)
    newdata=concatdf()
    y = newdata['Label']
    X = newdata.drop('Label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    # print(X_train.shape, X_test.shape)
    DT(X_train, X_test, y_train, y_test)
    RR(X_train, X_test, y_train, y_test)
    KNN(X_train, X_test, y_train, y_test)
