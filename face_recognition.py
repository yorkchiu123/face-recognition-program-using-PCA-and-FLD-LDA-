from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt



def read_img(path):
    im = Image.open(path)    # 讀取檔案
    im.show()    # 展示圖片
    # print(im.size)   # 輸出圖片大小


def img_to_vector(filename):
    img  = Image.open(filename)
    img_vector = np.array(img).flatten() # 轉為10304x1的向量
    # print(img_vector.size)

    return img_vector


def read_dataset(path):
    file = os.listdir(path) # 獲取path中所有檔案名稱(不含子資料夾)
    dataset = []
    label = []
    for i in file:
        filename = os.listdir(path+i) # 獲取子資料夾內所有檔案名稱
        for j in filename:
            img = img_to_vector(path+i+'/'+j).astype(np.int64) # 呼叫函式，讀取路徑檔案
            dataset.append(img) 
            label.append(i)
    # print(dataset)
    # print(label)

    return np.array(dataset), np.array(label)


def split_data(dataset, label):
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.5, random_state=42) # 將資料集切一半

    return X_train, X_test, y_train, y_test


def dimensionality_reduction_PCA(n_components, X_train):
    print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))

    # run randomized SVD by the method of Halko et al.
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    return pca

def dimensionality_reduction_LDA(n_components, X_train, y_train):
    print("Extracting the top %d fisherfaces from %d faces"% (n_components, X_train.shape[0]))
    pca = PCA(n_components=n_components).fit(X_train)
    lda = LDA().fit(pca.transform(X_train), y_train)

    return lda, pca


def train_text_transform_Model(model, X_train, X_test):
    print("Start Transform")
    X_train_model = model.transform(X_train)
    X_test_model = model.transform(X_test)

    return X_train_model, X_test_model


def train_text_transform_LDA(lda, pca, X_train, X_test):
    print("Start Transform")
    X_train_lda = lda.transform(pca.transform(X_train))
    X_test_lda = lda.transform(pca.transform(X_test))

    return X_train_lda, X_test_lda


def classification_svc(X_train_model, y_train):
    print("Fitting")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_model, y_train)

    return clf


def prediction(model, data):
    print("Start predict")
    y_pred = model.predict(data)

    return y_pred


def plot_confusion_matrix(y_true, y_pred, matrix_title):
    """confusion matrix computation and display"""
    plt.figure(figsize=(9, 9), dpi=100)

    # use sklearn confusion matrix
    cm_array = confusion_matrix(y_true, y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(matrix_title, fontsize=16)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))

    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    # read_img("face/s1/1.pgm")
    # img_to_vector("face/s1/1.pgm")

    path = "face/"
    dataset, label = read_dataset(path)
    X_train, X_test, y_train, y_test = split_data(dataset, label) # 將資料集切割為train與test
    

    components = [10, 20, 30 ,40, 50] # 所要降維的維度
    for n_components in components:
        # PCA
        print("-----------------PCA components: "+str(n_components)+" -----------------")
        pca= dimensionality_reduction_PCA(n_components, X_train)
        X_train_pca, X_test_pca = train_text_transform_Model(pca, X_train, X_test)
        classification = classification_svc(X_train_pca, y_train) # 訓練 SVM model
        y_pred = prediction(classification, X_test_pca) # 使用test data 來做預測
        # print(confusion_matrix(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, "matriz")
        print('維度%d: PCA識別率: %.2f%%' % (n_components, (y_pred == np.array(y_test)).mean() * 100))
        print("-----------------PCA components: "+str(n_components)+" -----------------")

        # LDA
        print("-----------------LDA components: "+str(n_components)+" -----------------")        
        lda, pca = dimensionality_reduction_LDA(n_components, X_train, y_train)
        X_train_lda, X_test_lda = train_text_transform_LDA(lda, pca, X_train, X_test)
        classification = classification_svc(X_train_lda, y_train)
        y_pred = prediction(classification, X_test_lda)
        plot_confusion_matrix(y_test, y_pred, "matriz")
        print('維度%d: LDA識別率: %.2f%%' % (n_components, (y_pred == np.array(y_test)).mean() * 100))
        print("-----------------LDA components: "+str(n_components)+" -----------------")        